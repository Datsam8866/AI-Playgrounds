import math
import unittest

import numpy as np

import evaluate_game_predictions_npb_regime as eval_npb
import predict_today_npb as predict_npb


class TestNpbPlatt(unittest.TestCase):
    def test_fit_platt_requires_enough_rows(self):
        probs = [0.55] * 49
        actuals = [0, 1] * 24 + [0]
        self.assertIsNone(eval_npb.fit_platt(probs, actuals))

    def test_apply_platt_ab_matches_sklearn_scaler(self):
        raw_probs = [0.22, 0.31, 0.38, 0.42, 0.47, 0.51, 0.56, 0.61, 0.68, 0.75] * 8
        actuals = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 8
        scaler = eval_npb.fit_platt(raw_probs, actuals)
        self.assertIsNotNone(scaler)

        a = float(scaler.coef_[0, 0])
        b = float(scaler.intercept_[0])
        sample_prob = 0.63

        expected = eval_npb.apply_platt(scaler, sample_prob)
        actual = predict_npb.apply_platt_ab(sample_prob, a, b)
        self.assertAlmostEqual(actual, expected, places=12)

    def test_logit_is_bounded(self):
        self.assertTrue(math.isfinite(eval_npb._logit(0.0)))
        self.assertTrue(math.isfinite(eval_npb._logit(1.0)))
        self.assertTrue(math.isfinite(predict_npb._logit(0.0)))
        self.assertTrue(math.isfinite(predict_npb._logit(1.0)))


if __name__ == "__main__":
    unittest.main()
