import io
import sqlite3
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nba_injury_scraper as scraper


class TestNbaInjuryScraper(unittest.TestCase):
    def test_fetch_and_store_injuries_returns_only_out_and_doubtful(self):
        conn = sqlite3.connect(":memory:")
        conn.executescript(scraper.SCHEMA_SQL)

        payloads = [
            {
                "sports": [
                    {
                        "leagues": [
                            {
                                "teams": [
                                    {"team": {"id": "1", "abbreviation": "LAL"}},
                                    {"team": {"id": "2", "abbreviation": "BOS"}},
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "injuries": [
                    {"athlete": {"displayName": "LeBron James"}, "type": {"name": "Out"}},
                    {"athlete": {"displayName": "Anthony Davis"}, "type": {"name": "Questionable"}},
                ]
            },
            {},
        ]

        def fake_fetch_json(_url):
            return payloads.pop(0)

        stdout = io.StringIO()
        with patch.object(scraper, "fetch_json", side_effect=fake_fetch_json), patch.object(scraper.time, "sleep"), redirect_stdout(stdout):
            injury_map = scraper.fetch_and_store_injuries(conn, "2026-05-15")

        rows = conn.execute(
            "SELECT player_name, team_abbr, status, scraped_date FROM player_injuries ORDER BY team_abbr, player_name"
        ).fetchall()

        self.assertEqual(injury_map, {"LAL": ["LeBron James"]})
        self.assertEqual(
            rows,
            [
                ("Anthony Davis", "LAL", "questionable", "2026-05-15"),
                ("LeBron James", "LAL", "out", "2026-05-15"),
            ],
        )
        self.assertIn("No injuries reported for BOS", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
