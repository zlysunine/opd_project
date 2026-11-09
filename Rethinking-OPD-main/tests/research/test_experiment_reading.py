"""Question-cluster uncertainty and immutable archive report checks."""
from pathlib import Path
import unittest

import numpy as np

from scripts.research.build_experiment_reading import audit, bootstrap_change, exact_mcnemar, paired_ids, truth


class ExperimentReadingTests(unittest.TestCase):
    def test_token_weighted_ratio_and_paired_resampling(self):
        indices=np.asarray([[0,0],[1,1],[0,1]])
        result=bootstrap_change([1,9],[1,3],[2,6],[1,3],indices)
        self.assertEqual(result['initial'],2.5)  # Question-equal mean would be 2.
        self.assertEqual(result['final'],2.)
        self.assertEqual(result['change'],-.5)
        np.testing.assert_allclose(result['ci95'],np.quantile([1.,-1.,-.5],[.025,.975]))

    def test_each_model_step_retains_its_own_denominator(self):
        result=bootstrap_change([1,9],[1,3],[4,1],[2,1],np.asarray([[0,1]]))
        self.assertAlmostEqual(result['change'],5/3-2.5)
        self.assertEqual(result['initial_tokens'],4)
        self.assertEqual(result['final_tokens'],3)

    def test_empty_intersection_cluster_not_invented_zero(self):
        result=bootstrap_change([0,2],[0,2],[0,4],[0,2],np.asarray([[0,0],[1,1],[0,1]]))
        self.assertEqual(result['valid_bootstrap_draws'],2)
        np.testing.assert_equal(result['ci95'],[1.,1.])
        with self.assertRaises(ValueError):
            bootstrap_change([1],[-1],[2],[1],np.asarray([[0]]))

    def test_exact_mcnemar_and_boolean_transport(self):
        self.assertEqual(exact_mcnemar(4,6),.75390625)
        self.assertEqual(exact_mcnemar(4,6),exact_mcnemar(6,4))
        self.assertEqual(exact_mcnemar(0,0),1.)
        self.assertFalse(truth('False'))
        self.assertTrue(truth('True'))
        with self.assertRaises(ValueError):truth('missing')
        with self.assertRaises(ValueError):paired_ids({'q1':1},{'q2':1})

    def test_archived_second_run_totals_and_aggregation(self):
        source=Path(__file__).resolve().parents[2]/'results/20261028_qwen17b_opsd20'
        s,_=audit(source,replicates=100,seed=7)
        self.assertEqual(s['integrity']['checks']['distribution_aggregates_checked'],1040)
        self.assertEqual(s['integrity']['checks']['curvature_aggregates_checked'],560)
        self.assertEqual(s['behavior']['heldout_probe']['gained'],4)
        self.assertEqual(s['behavior']['heldout_probe']['lost'],6)
        self.assertEqual(s['config']['monitor']['quality_gate']['max_extraction_failure_rate'],.3)
        self.assertAlmostEqual(s['distributions']['heldout_probe']['fixed_base_rollout']['changes']
                               ['kl_student_teacher_nats']['final'],.03943236844936175)
        self.assertEqual(s['representation']['heldout_probe']['student_teacher']['rsa_rho_student_teacher']['increased'],51)
        self.assertEqual(s['representation']['heldout_probe']['student_teacher']['rsa_rho_student_teacher']['decreased'],33)


if __name__=='__main__':unittest.main()
