import unittest
from collections import defaultdict, deque
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

    def test_compute_injury_pts_sums_known_players_only(self):
        player_season_avg = {
            ("LAL", "LeBron James"): 25.0,
            ("LAL", "Anthony Davis"): 24.0,
        }

        total = builder.compute_injury_pts(
            "LAL",
            ["LeBron James", "Unknown Player"],
            player_season_avg,
        )

        self.assertEqual(total, 25.0)

    def test_load_injury_map_filters_out_and_doubtful(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """
            CREATE TABLE player_injuries (
                team_abbr TEXT,
                player_name TEXT,
                status TEXT,
                scraped_date TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO player_injuries(team_abbr, player_name, status, scraped_date) VALUES (?, ?, ?, ?)",
            [
                ("LAL", "LeBron James", "out", "2026-05-15"),
                ("LAL", "Anthony Davis", "questionable", "2026-05-15"),
                ("BOS", "Jrue Holiday", "doubtful", "2026-05-15"),
            ],
        )

        injury_map = builder.load_injury_map(conn, "2026-05-15")

        self.assertEqual(injury_map, {"LAL": ["LeBron James"], "BOS": ["Jrue Holiday"]})

    def test_to_feature_row_writes_injury_fields(self):
        game = {
            "game_id": "002",
            "season_year": 2025,
            "game_date": "2026-05-15",
            "home_team_id": 1,
            "vis_team_id": 2,
            "home_team_abbr": "LAL",
            "vis_team_abbr": "BOS",
            "home_win": 1,
        }

        row = builder.to_feature_row(
            game=game,
            team_history=defaultdict(lambda: deque(maxlen=builder.WINDOW)),
            elo_by_team=defaultdict(lambda: builder.ELO_BASE),
            streak_by_team={},
            last_game_dates={},
            season_games=defaultdict(int),
            neutral=0,
            injury_map={"LAL": ["LeBron James"], "BOS": ["Jrue Holiday"]},
            player_season_avg={
                ("LAL", "LeBron James"): 25.0,
                ("BOS", "Jrue Holiday"): 18.0,
            },
        )

        self.assertEqual(row["home_injury_pts"], 25.0)
        self.assertEqual(row["vis_injury_pts"], 18.0)
        self.assertEqual(row["diff_injury_pts"], 7.0)


if __name__ == "__main__":
    unittest.main()
