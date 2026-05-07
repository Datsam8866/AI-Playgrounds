import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NPB_DIR = ROOT / "NPB"
sys.path.insert(0, str(NPB_DIR))

import evaluate_game_predictions_npb_regime as evaluate_npb  # noqa: E402
import predict_today_npb  # noqa: E402


def _mature_row(sp_available):
    return {
        "home_season_games_before": 20,
        "vis_season_games_before": 20,
        "sp_available": sp_available,
    }


def _early_row():
    return {
        "home_season_games_before": 3,
        "vis_season_games_before": 20,
        "sp_available": 1,
    }


def test_npb_predict_routes_fallback_when_starter_features_are_unavailable():
    assert predict_today_npb.route_regime(_early_row()) == "early"
    assert predict_today_npb.route_regime(_mature_row(0)) == "fallback"
    assert predict_today_npb.route_regime(_mature_row(None)) == "fallback"
    assert predict_today_npb.route_regime(_mature_row(1)) == "primary"


def test_npb_evaluation_routes_fallback_when_starter_features_are_unavailable():
    assert evaluate_npb.route_regime(_early_row()) == "early_baseline"
    assert evaluate_npb.route_regime(_mature_row(0)) == "fallback"
    assert evaluate_npb.route_regime(_mature_row(None)) == "fallback"
    assert evaluate_npb.route_regime(_mature_row(1)) == "primary"
