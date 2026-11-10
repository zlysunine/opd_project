"""Full-vocabulary diagnostics; sequence statistics use all masked response tokens."""
import numpy as np
import torch


def selected_log_probs(model, hidden, indices, chunk_size=16):
    if not indices:
        return np.empty((0, model.config.vocab_size), dtype=np.float32)
    head = model.get_output_embeddings()
    result = []
    for start in range(0, len(indices), chunk_size):
        selected = hidden[indices[start:start+chunk_size]]
        result.append(head(selected).float().log_softmax(-1).cpu().numpy())
    return np.concatenate(result)


def nll_bits(model, hidden, response, prediction_indices, chunk_size=16):
    if not response:
        return np.nan
    total = 0.
    for start in range(0, len(response), chunk_size):
        ix = prediction_indices[start:start+chunk_size]
        logp = model.get_output_embeddings()(hidden[ix]).float().log_softmax(-1)
        targets = torch.tensor(response[start:start+chunk_size], device=logp.device)
        total -= logp.gather(-1, targets[:, None]).sum().item()
    return total/len(response)/np.log(2.)


def distribution_metrics(logp, logq, initial=None, ks=(10,50,100)):
    """Single-state diagnostic retained for numerical tests, not sequence averaging."""
    p, q = np.exp(logp.astype(np.float64)), np.exp(logq.astype(np.float64))
    result = dict(student_entropy_nats=float(-(p*logp).sum()), teacher_entropy_nats=float(-(q*logq).sum()),
                  kl_student_teacher_nats=float((p*(logp-logq)).sum()))
    result['entropy_gap_signed_nats'] = result['teacher_entropy_nats']-result['student_entropy_nats']
    result['entropy_gap_abs_nats'] = abs(result['entropy_gap_signed_nats'])
    if initial is not None:
        result['kl_to_initial_nats'] = float((p*(logp-initial)).sum())
    # Stable full sort: tied probabilities use token ID. No support renormalization.
    a, b = np.argsort(-logp, kind='stable'), np.argsort(-logq, kind='stable')
    for k in ks:
        if not 1 <= k <= len(p):
            for suffix in ('overlap','shared_mass_student','shared_mass_teacher'):
                result[f'token_top{k}_{suffix}'] = np.nan
            continue
        shared = np.intersect1d(a[:k], b[:k])
        result[f'token_top{k}_overlap'] = len(shared)/k
        result[f'token_top{k}_shared_mass_student'] = float(p[shared].sum())
        result[f'token_top{k}_shared_mass_teacher'] = float(q[shared].sum())
    return result


def pair_token_metrics(logp, logq, ks=(16,), primary_k=16):
    """Per-position full-vocab entropy and online-compatible overlap/OTA.

    Inputs are [positions, vocabulary] log probabilities at identical prefixes.
    Top-k selection follows torch.topk, as in the online worker. Empty-intersection
    OTA is NaN and must have its own denominator; other metrics retain that token.
    """
    if logp.shape != logq.shape or logp.ndim != 2:
        raise ValueError('Expected aligned [positions, vocabulary] distributions')
    if not torch.isfinite(logp).all() or not torch.isfinite(logq).all():
        raise ValueError('Nonfinite full-vocabulary log probabilities')
    ks=sorted(set([primary_k,*ks]))
    if any(isinstance(k,bool) or not isinstance(k,int) or not 1<=k<=logp.shape[-1] for k in ks):
        raise ValueError('Distribution top-k must fit the full vocabulary')
    p,q=logp.exp(),logq.exp()
    hp,hq=-(p*logp).sum(-1),-(q*logq).sum(-1)
    values=dict(student_entropy_nats=hp,teacher_entropy_nats=hq,
                entropy_gap_signed_nats=hq-hp,entropy_gap_abs_nats=(hq-hp).abs(),
                kl_student_teacher_nats=(p*(logp-logq)).sum(-1),
                kl_teacher_student_nats=(q*(logq-logp)).sum(-1))
    for k in ks:
        # Select each k independently to match online torch.topk even with ties.
        ids=torch.topk(logp,k,dim=-1).indices
        teacher_ids=torch.topk(logq,k,dim=-1).indices
        shared=(ids.unsqueeze(-1)==teacher_ids.unsqueeze(-2)).any(-1)
        count=shared.sum(-1)
        slp=logp.gather(-1,ids)
        tlp=logq.gather(-1,ids)
        advantage=torch.full_like(hp,float('nan'))
        nonempty=count>0
        if nonempty.any():
            selected=shared[nonempty]
            sp=slp[nonempty].masked_fill(~selected,-torch.inf)
            tq=tlp[nonempty].masked_fill(~selected,-torch.inf)
            sp=sp-torch.logsumexp(sp,-1,keepdim=True)
            tq=tq-torch.logsumexp(tq,-1,keepdim=True)
            ratio=torch.where(selected,tq-sp,0.)
            advantage[nonempty]=(sp.exp()*ratio).sum(-1)/count[nonempty]
        values.update({f'token_top{k}_overlap':count.to(hp.dtype)/k,
                       f'token_top{k}_shared_mass_student':(slp.exp()*shared).sum(-1),
                       f'token_top{k}_shared_mass_teacher':(tlp.exp()*shared).sum(-1),
                       f'token_top{k}_empty_intersection_rate':(~nonempty).to(hp.dtype),
                       f'token_top{k}_overlap_token_advantage_nats':advantage})
    values['overlap_token_advantage_nats']=values[f'token_top{primary_k}_overlap_token_advantage_nats']
    return values


class TokenAccumulator:
    """Accumulate sums/counts, never averages of unequal-length answers or chunks."""
    def __init__(self):
        self.names=None
        self.sums=None
        self.counts=None

    def update(self,values,mask):
        names=sorted(values)
        matrix=torch.stack([values[name] for name in names],-1)
        if mask.shape != matrix.shape[:1]:
            raise ValueError('One response-mask value per prediction position required')
        if self.names is not None and self.names!=names:
            raise ValueError('Token metric schema changed inside a trajectory')
        valid=mask.bool()[:,None]&torch.isfinite(matrix)
        # Only intersection OTA may be undefined at a valid response token.
        required=[i for i,name in enumerate(names) if 'advantage' not in name]
        if not torch.isfinite(matrix[mask.bool()][:,required]).all():
            raise ValueError('Undefined entropy/KL at a valid token')
        sums=torch.where(valid,matrix,0.).double().sum(0)
        counts=valid.sum(0)
        if self.names is None:
            self.names,self.sums,self.counts=names,sums,counts
        else:
            self.sums+=sums
            self.counts+=counts

    def result(self):
        if self.names is None:
            return {},{},{}
        sums=self.sums.cpu().tolist()
        counts=self.counts.cpu().tolist()
        return ({name:total/n if n else None for name,total,n in zip(self.names,sums,counts)},
                dict(zip(self.names,sums)),dict(zip(self.names,counts)))
