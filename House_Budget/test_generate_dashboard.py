import sqlite3
import tempfile
import unittest
from pathlib import Path

import generate_dashboard as dashboard


class DashboardDataTests(unittest.TestCase):
    def make_db(self):
        tmp = tempfile.TemporaryDirectory()
        db_path = Path(tmp.name) / "budget.db"
        con = sqlite3.connect(db_path)
        con.execute(
            """
            CREATE TABLE transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                date TEXT,
                category TEXT,
                item TEXT,
                amount REAL,
                currency TEXT,
                amount_twd REAL,
                who TEXT,
                note TEXT
            )
            """
        )
        rows = [
            ("household_2026/04", "2026-04-01", "Income", "Sam Income", 35000, "TWD", 35000, "Sam", None),
            ("household_2026/04", "2026-04-02", "Trasportation", "Parking", -30, "TWD", -30, "Sam", None),
            ("household_2026/04", "2026-04-03", "Food", "Dinner", -500, "TWD", -500, "Rita", None),
            ("household_2026/05", "2026-05-01", "Income", "Rita Income", 27500, "TWD", 27500, "Rita", None),
            ("household_2026/05", "2026-05-02", "Transportation", "Parking", -18, "TWD", -18, "Sam", None),
            ("household_2026/05", "2026-05-03", "Accommodation", "Rent", -29305, "TWD", -29305, "Sam", None),
            ("au_travel/Budget", "2026-05-04", "Accommdation", "Hotel", -100, "AUD", -2100, "Sam", None),
        ]
        con.executemany(
            """
            INSERT INTO transactions
                (source, date, category, item, amount, currency, amount_twd, who, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
        con.close()
        self.addCleanup(tmp.cleanup)
        return db_path

    def test_normalizes_common_category_variants(self):
        self.assertEqual(dashboard.normalize_category("Trasportation"), "Transportation")
        self.assertEqual(dashboard.normalize_category("Accommdation"), "Accommodation")

    def test_monthly_kpis_include_income_expense_net_and_delta(self):
        data = dashboard.build_dashboard_data(self.make_db())
        may = data["months"]["2026-05"]

        self.assertEqual(may["income"], 27500)
        self.assertEqual(may["expense"], -31423)
        self.assertEqual(may["net"], -3923)
        self.assertEqual(may["expense_delta"], -30893)

    def test_large_expenses_are_sorted_by_absolute_amount(self):
        data = dashboard.build_dashboard_data(self.make_db())
        large = data["months"]["2026-05"]["large_expenses"]

        self.assertEqual(large[0]["item"], "Rent")
        self.assertEqual(large[0]["amount_twd"], -29305)
        self.assertEqual(large[1]["item"], "Hotel")


if __name__ == "__main__":
    unittest.main()
