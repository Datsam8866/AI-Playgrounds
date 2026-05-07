from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_update_all_uses_playsport_for_cpbl():
    script = (ROOT / "update_all.ps1").read_text(encoding="utf-8")
    cpbl_block = script.split("# --- CPBL ---", 1)[1].split("# --- NPB", 1)[0]

    assert "playsport_results_sync.py" in cpbl_block
    assert "--league cpbl" in cpbl_block
    assert "cpbl_boxscore_scraper.py" not in cpbl_block


def test_cpbl_dashboard_refresh_uses_playsport_sync():
    adapter = (ROOT / "CPBL" / "run_dashboard.py").read_text(encoding="utf-8")
    refresh_body = adapter.split("def _run_refresh", 1)[1].split("from cpbl_betting_ev", 1)[0]

    assert "playsport_results_sync.py" in refresh_body
    assert "--league" in refresh_body
    assert "cpbl_boxscore_scraper.py" not in refresh_body
