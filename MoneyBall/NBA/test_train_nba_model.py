import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_nba_model as train_nba


class TestTrainNbaModel(unittest.TestCase):
    def test_rolling_train_years_uses_five_seasons(self):
        self.assertEqual(train_nba.rolling_train_years(2016), [2011, 2012, 2013, 2014, 2015])
        self.assertEqual(train_nba.rolling_train_years(2025), [2020, 2021, 2022, 2023, 2024])

    def test_fit_calibrator_requires_at_least_fifty_rows(self):
        probs = [0.55] * 49
        actuals = ([0, 1] * 24) + [0]
        self.assertIsNone(train_nba.fit_calibrator(probs, actuals))

    def test_features_replace_raw_elo_columns_with_neutral_site_flag(self):
        self.assertNotIn("home_elo", train_nba.FEATURES)
        self.assertNotIn("vis_elo", train_nba.FEATURES)
        self.assertIn("is_neutral_site", train_nba.FEATURES)
        self.assertIn("home_injury_pts", train_nba.FEATURES)
        self.assertIn("vis_injury_pts", train_nba.FEATURES)
        self.assertIn("diff_injury_pts", train_nba.FEATURES)
        self.assertEqual(len(train_nba.FEATURES), 31)

    def test_matrix_from_rows_uses_training_medians_for_nulls(self):
        rows = [
            {"diff_elo": 12.0, "home_rest": 2.0},
            {"diff_elo": None, "home_rest": 1.0},
            {"diff_elo": 18.0, "home_rest": None},
        ]
        medians = train_nba.compute_feature_medians(rows, ["diff_elo", "home_rest"])
        matrix = train_nba.matrix_from_rows(rows, ["diff_elo", "home_rest"], medians)

        np.testing.assert_allclose(medians, np.array([15.0, 1.5]))
        np.testing.assert_allclose(
            matrix,
            np.array(
                [
                    [12.0, 2.0],
                    [15.0, 1.0],
                    [18.0, 1.5],
                ]
            ),
        )

    def test_apply_calibrator_returns_original_prob_when_calibrator_missing(self):
        self.assertAlmostEqual(train_nba.apply_calibrator(None, 0.63), 0.63)

    def test_brier_score_matches_mean_squared_error(self):
        score = train_nba.brier_score([0.8, 0.2], [1, 0])
        self.assertAlmostEqual(score, 0.04)

    def test_expected_calibration_error_uses_weighted_bin_gap(self):
        ece = train_nba.expected_calibration_error([0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1], n_bins=4)
        self.assertAlmostEqual(ece, 0.3)


if __name__ == "__main__":
    unittest.main()
