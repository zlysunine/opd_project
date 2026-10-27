"""Report summaries preserve missing values, sample counts and metric coverage."""
from collections import defaultdict
import unittest

from scripts.research.build_metric_report import (
    POINT_COLUMNS, check_measurement_fields, sample_entries, scalar_point,
)


class MetricReportTests(unittest.TestCase):
    def test_missing_scalar_stays_missing_instead_of_zero(self):
        point=scalar_point(5,None,float('nan'),0,8,'undefined')
        self.assertEqual(len(point),len(POINT_COLUMNS))
        self.assertIsNone(point[2])
        self.assertEqual(point[3:5],[0,8])
        self.assertEqual(point[-1],'undefined')

    def test_per_sample_histogram_counts_only_finite_observations(self):
        groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        context=(('split','heldout_probe'),)
        groups['value'][context][(20,None)]=[0.,2.,None,float('nan')]
        metric=sample_entries(groups,'behavior')[0]
        point=metric['variants'][0]['points'][0]
        self.assertEqual(point[2],1.)
        self.assertEqual(point[3:5],[2,4])
        self.assertEqual(point[7],1.)
        self.assertEqual(sum(point[13]),2)
        self.assertEqual(len(point),len(POINT_COLUMNS))

    def test_wholly_undefined_context_has_no_fabricated_average(self):
        groups=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
        groups['surprisal_bits'][(('role','privileged_teacher'),)][(20,28)]=[None,None]
        point=sample_entries(groups,'curvature')[0]['variants'][0]['points'][0]
        self.assertIsNone(point[2])
        self.assertEqual(point[3:5],[0,2])
        self.assertEqual(sum(point[13]),0)
        self.assertEqual(point[-1],'undefined')

    def test_new_numeric_measurement_cannot_be_silently_omitted(self):
        check_measurement_fields({'layer':28,'mean_deg':12.},['mean_deg'])
        check_measurement_fields(dict(response_position='all_valid_tokens',n_valid_tokens=16384,
                                      response_length=16384,student_entropy_nats=1.,
                                      metric_sums={'student_entropy_nats':16384.},
                                      metric_token_counts={'student_entropy_nats':16384}),['student_entropy_nats'])
        with self.assertRaisesRegex(ValueError,'new_metric'):
            check_measurement_fields({'mean_deg':12.,'new_metric':.4},['mean_deg'])


if __name__=='__main__':unittest.main()
