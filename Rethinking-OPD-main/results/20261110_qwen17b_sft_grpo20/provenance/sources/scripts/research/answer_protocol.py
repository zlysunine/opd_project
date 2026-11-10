"""One answer contract for Base prompts, dataset labels, rewards and evaluation.

Extraction never reads ground truth. Legacy math equivalence is reused only on
already-extracted answers; its boxed-only response extractor is not called.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import re


VERSION = 'opsd_answer_v2'
INSTRUCTIONS = (
    'Solve the problem carefully.\n'
    'Finish your response with exactly one final-answer line:\n'
    'Final answer: \\boxed{YOUR_ANSWER}\n\n'
    'Inside the box, write only the final answer, without explanation.\n'
    'For true/false questions, write True or False.\n'
    'For multiple-choice questions, write the option label given in the problem.\n'
    'Do not write anything after the final-answer line.'
)
_FINAL = re.compile(r'(?im)^[ \t]*(?:final\s+answer|最终答案)[ \t]*[:：][ \t]*')
_BOX = re.compile(r'\\(boxed|fbox)\s*\{')
_BOOL = {'true': 'True', 'false': 'False', 'yes': 'True', 'no': 'False', '真': 'True', '假': 'False',
         '正确': 'True', '错误': 'False'}


def build_student_prompt(question):
    return f'Problem:\n{question}\n\nInstructions:\n{INSTRUCTIONS}\n\nSolution:\n'


def _clean(value):
    value = str(value).strip()
    for left, right in [('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')]:
        if value.startswith(left) and value.endswith(right) and len(value) >= len(left)+len(right):
            value = value[len(left):-len(right)].strip()
    return value


def _comparison_text(value):
    return re.sub(r'\s+', '', _clean(value)).replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')


def _boolean(value):
    value=_clean(value)
    wrapped=re.fullmatch(r'\\(?:text|mathrm)\s*\{([^{}]+)\}',value)
    if wrapped:
        value=wrapped.group(1).strip()
    return _BOOL.get(value.casefold())


def _boxes(text):
    records = []
    for match in _BOX.finditer(text):
        depth, closing = 1, None
        for i in range(match.end(), len(text)):
            # Escaped \{ and \} are mathematical literals, not grouping braces.
            slashes, j = 0, i-1
            while j >= 0 and text[j] == '\\':
                slashes, j = slashes+1, j-1
            if slashes % 2:
                continue
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    closing = i
                    break
        records.append(dict(start=match.start(), end=None if closing is None else closing+1,
                            command=match.group(1),
                            answer=None if closing is None else _clean(text[match.end():closing])))
    return records


def extract_final_answer(text):
    """Return a candidate and an auditable status; never guess the last number."""
    def result(status, answer=None, source=None, compliant=False):
        return dict(extracted_answer=answer, extraction_failed=status != 'extracted',
                    parse_status=status, extraction_source=source, format_compliant=compliant,
                    answer_protocol_version=VERSION)

    if not isinstance(text, str):
        return result('invalid_response_type')
    finals = list(_FINAL.finditer(text))
    sections = [text[m.end():finals[i+1].start() if i+1 < len(finals) else len(text)]
                for i, m in enumerate(finals)] if finals else [text]
    parsed = []
    for section in sections:
        boxes = _boxes(section)
        # A trailing malformed command must not expose an earlier intermediate answer.
        last_command = list(re.finditer(r'\\(?:boxed|fbox)\b', section))
        if last_command and (not boxes or last_command[-1].start() != boxes[-1]['start']):
            return result('malformed_box')
        if boxes:
            box = boxes[-1]
            if box['end'] is None:
                return result('unclosed_box')
            answer = box['answer']
            if finals:
                if section[:box['start']].strip() not in ('', '$', '$$', r'\(', r'\['):
                    return result('ambiguous_final_region')
                source = 'final_boxed' if box['command'] == 'boxed' else 'final_fbox'
                trailing = section[box['end']:].strip()
                compliant = len(finals) == 1 and source == 'final_boxed' and trailing in ('', '$', '$$', r'\)', r'\]')
            else:
                source, compliant = 'legacy_boxed', False
        elif finals:
            meaningful = [line.strip() for line in section.splitlines() if line.strip()]
            if len(meaningful) > 1:
                return result('ambiguous_final_region')
            answer, source, compliant = _clean(meaningful[0]) if meaningful else '', 'final_plain', False
        else:
            return result('missing_final_answer')
        if not answer or not answer.strip():
            return result('empty_answer', source=source)
        if re.fullmatch(r"(?i)(?:your[_ ]answer|unknown|i don['’]t know|n/?a)", answer.strip()):
            return result('non_answer', source=source)
        parsed.append((answer, source, compliant))
    if len({_comparison_text(answer) for answer, _, _ in parsed}) > 1:
        return result('ambiguous_answers')
    answer, source, compliant = parsed[-1]
    return result('extracted', answer, source, compliant)


def answer_metadata(row):
    """Conservative label typing. Never infer A=True from a reference solution."""
    question = str(row.get('question', ''))
    truth = row.get('ground_truth', row.get('final_answer'))
    if truth is None or not str(truth).strip():
        return dict(answer_encoding_status='missing_ground_truth')
    truth = str(truth).strip()
    if '\\boxed' in truth or _FINAL.search(truth):
        parsed = extract_final_answer(truth)
        if parsed['extraction_failed']:
            return dict(answer_encoding_status='malformed_ground_truth')
        truth = parsed['extracted_answer']
    mapping = row.get('option_mapping') or row.get('choices') or {}
    if not isinstance(mapping, dict) or any(not isinstance(v, (str, bool, int, float)) for v in mapping.values()):
        return dict(answer_encoding_status='invalid_option_mapping')
    mapping = {str(k).strip().upper(): str(v).strip() for k, v in mapping.items()}
    # Only visibly labelled options are inferred, not a hidden A/B convention.
    if not mapping:
        options = re.findall(r'(?m)^\s*(?:\(([A-H])\)|([A-H])[.)])\s+(.+)$', question)
        mapping = {a or b: value.strip() for a, b, value in options}
        if len(mapping) < 2:
            mapping = {}
    boolean_question = bool(re.search(r'(?i)\btrue\s*(?:or|/)\s*false\b|判断.{0,30}[真假]', question))
    kind = row.get('answer_type') or ('boolean' if boolean_question or _boolean(truth) is not None
                                    else 'multiple_choice' if mapping else 'math')
    if kind not in ('math', 'boolean', 'multiple_choice'):
        return dict(answer_encoding_status='unsupported_answer_type')
    if boolean_question and kind != 'boolean':
        return dict(answer_encoding_status='conflicting_answer_type')
    if kind == 'boolean':
        value = mapping.get(truth.upper(), truth)
        canonical = _boolean(value)
        if canonical is None:
            return dict(answer_encoding_status='unmapped_boolean_label')
    elif kind == 'multiple_choice':
        matches=[key for key,value in mapping.items() if _comparison_text(value)==_comparison_text(truth)]
        if not mapping or (truth.upper() not in mapping and len(matches)!=1):
            return dict(answer_encoding_status='missing_choice_mapping')
        canonical = truth.upper() if truth.upper() in mapping else matches[0]
    else:
        if not row.get('answer_type') and re.fullmatch('[A-H]', truth):
            return dict(answer_encoding_status='ambiguous_letter_answer')
        canonical = truth
    return dict(answer_encoding_status='ok', answer_type=kind, option_mapping=mapping,
                canonical_ground_truth=canonical, answer_protocol_version=VERSION)


@lru_cache(maxsize=1)
def _math_verifier():
    path = Path(__file__).resolve().parents[1]/'val/eval/utils.py'
    spec = importlib.util.spec_from_file_location('_opsd_math_equivalence_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grade_extracted_answer(candidate, ground_truth, metadata=None):
    """Compare candidates directly; no second boxed-only response extraction."""
    metadata = metadata or answer_metadata(dict(ground_truth=ground_truth))
    if metadata.get('answer_encoding_status') != 'ok':
        return False
    candidate = _clean(candidate)
    target = metadata['canonical_ground_truth']
    if not candidate:
        return False
    kind, mapping = metadata['answer_type'], metadata['option_mapping']
    if kind == 'boolean':
        return _boolean(mapping.get(candidate.upper(), candidate)) == target
    if kind == 'multiple_choice':
        if candidate.upper() in mapping:
            return candidate.upper() == target
        labels = [key for key, value in mapping.items() if _comparison_text(value) == _comparison_text(candidate)]
        return len(labels) == 1 and labels[0] == target
    if _comparison_text(candidate) == _comparison_text(target):
        return True
    # Exact arithmetic also keeps integer fixtures independent of LaTeX packages.
    numeric = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    if re.fullmatch(numeric, candidate) and re.fullmatch(numeric, target):
        from decimal import Decimal
        return Decimal(candidate) == Decimal(target)
    module = _math_verifier()
    return bool(module.grade_answer_mathd(candidate, target) or module.grade_answer_sympy(candidate, target))


def evaluate_answer(text, ground_truth, truncated=False, row=None):
    parsed = extract_final_answer(text)
    metadata = answer_metadata({**(row or {}), 'ground_truth': ground_truth})
    parsed.update(answer_encoding_status=metadata['answer_encoding_status'],
                  answer_type=metadata.get('answer_type'), verifier_revision=VERSION)
    if not parsed['extraction_failed'] and metadata.get('answer_type') == 'boolean':
        value = metadata['option_mapping'].get(parsed['extracted_answer'].upper(), parsed['extracted_answer'])
        if _boolean(value) is None:
            parsed.update(extraction_failed=True, parse_status='invalid_answer_type')
    parsed['correct'] = bool(not truncated and not parsed['extraction_failed'] and
                             grade_extracted_answer(parsed['extracted_answer'], ground_truth, metadata))
    return parsed
