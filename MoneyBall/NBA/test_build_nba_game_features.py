import unittest
from collections import defaultdict, deque

import build_nba_game_features as builder


class TestBuildNbaGameFeatures(unittest.TestCase):
    def test_is_neutral_site_flags_2020_bubble_games(self):
        bubble_game = {"season_year": 2019, "game_date": "2020-07-30"}
        normal_game = {"season_year": 2019, "game_date": "2020-07-29"}
        next_season_game = {"season_year": 2020, "game_date": "2021-01-01"}

        self.assertEqual(builder.is_neutral_site(bubble_game), 1)
        self.assertEqual(builder.is_neutral_site(normal_game), 0)
        self.assertEqual(builder.is_neutral_site(next_season_game), 0)

    def test_elo_win_prob_drops_home_advantage_on_neutral_site(self):
        home_prob = builder.elo_win_prob(1500.0, 1500.0, neutral=0)
        neutral_prob = builder.elo_win_prob(1500.0, 1500.0, neutral=1)

        self.assertGreater(home_prob, neutral_prob)
        self.assertAlmostEqual(neutral_prob, 0.5)

    def test_to_feature_row_writes_neutral_flag(self):
        game = {
            "game_id": "001",
            "season_year": 2019,
            "game_date": "2020-08-01",
            "home_team_id": 1,
            "vis_team_id": 2,
            "home_team_abbr": "LAL",
            "vis_team_abbr": "LAC",
            "home_win": 1,
        }

        row = builder.to_feature_row(
            game=game,
            team_history=defaultdict(lambda: deque(maxlen=builder.WINDOW)),
            elo_by_team=defaultdict(lambda: builder.ELO_BASE),
            streak_by_team={},
            last_game_dates={},
            season_games=defaultdict(int),
            neutral=1,
        )

        self.assertEqual(row["is_neutral_site"], 1)
        self.assertAlmostEqual(row["elo_win_prob"], 0.5)


if __name__ == "__main__":
    unittest.main()
