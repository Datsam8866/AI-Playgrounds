PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    note TEXT
);

CREATE TABLE IF NOT EXISTS raw_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_sha256 TEXT,
    source_mtime TEXT,
    source_size_bytes INTEGER,
    metadata_json TEXT,
    FOREIGN KEY (run_id) REFERENCES ingestion_runs(run_id)
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_code TEXT NOT NULL UNIQUE,
    team_name TEXT NOT NULL,
    roster_name TEXT NOT NULL,
    aliases_json TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    acnt TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    roster_name TEXT,
    name_marker TEXT,
    jersey_number TEXT,
    position_name TEXT,
    bats_throws TEXT,
    height_weight TEXT,
    birthday TEXT,
    debut_date TEXT,
    school TEXT,
    draft_info TEXT,
    player_url TEXT,
    current_team_id INTEGER NOT NULL,
    FOREIGN KEY (current_team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_batting_season_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season_year INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    kind_name TEXT NOT NULL,
    total_team_games REAL,
    total_games REAL,
    plate_appearances REAL,
    avg REAL,
    obp REAL,
    slg REAL,
    ops REAL,
    home_run_cnt REAL,
    strike_out_cnt REAL,
    bases_on_balls_cnt REAL,
    wrc_plus REAL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_batting_career_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    kind_name TEXT NOT NULL,
    total_games REAL,
    plate_appearances REAL,
    avg REAL,
    obp REAL,
    slg REAL,
    ops REAL,
    home_run_cnt REAL,
    strike_out_cnt REAL,
    bases_on_balls_cnt REAL,
    wrc_plus REAL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_pitching_season_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season_year INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    kind_name TEXT NOT NULL,
    total_team_games REAL,
    total_games REAL,
    pitch_starting REAL,
    wins REAL,
    loses REAL,
    save_ok REAL,
    relief_point_cnt REAL,
    inning_pitched REAL,
    era REAL,
    whip REAL,
    pitch_cnt REAL,
    strike_out_cnt REAL,
    bases_on_balls_cnt REAL,
    home_run_cnt REAL,
    avg REAL,
    k9 REAL,
    b9 REAL,
    h9 REAL,
    fip REAL,
    era_plus REAL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_pitching_career_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    kind_name TEXT NOT NULL,
    total_games REAL,
    pitch_starting REAL,
    wins REAL,
    loses REAL,
    save_ok REAL,
    relief_point_cnt REAL,
    inning_pitched REAL,
    era REAL,
    whip REAL,
    strike_out_cnt REAL,
    bases_on_balls_cnt REAL,
    home_run_cnt REAL,
    avg REAL,
    k9 REAL,
    b9 REAL,
    h9 REAL,
    fip REAL,
    era_plus REAL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS player_fielding_season_stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    season_year INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    kind_name TEXT NOT NULL,
    defend_station_name TEXT,
    total_year_games REAL,
    total_games REAL,
    defend_cnt REAL,
    putout_cnt REAL,
    assist_cnt REAL,
    error_cnt REAL,
    join_double_play_cnt REAL,
    join_tripple_play_cnt REAL,
    passed_ball_cnt REAL,
    caught_stealing_cnt REAL,
    steal_cnt REAL,
    cs_pct REAL,
    fpct REAL,
    raw_json TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS team_season_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    historical_team_name TEXT NOT NULL,
    season_year INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    kind_name TEXT NOT NULL,
    section_name TEXT NOT NULL,
    rank_text TEXT,
    games REAL,
    wins REAL,
    ties REAL,
    losses REAL,
    win_pct REAL,
    games_back TEXT,
    home_record TEXT,
    away_record TEXT,
    extras_json TEXT,
    source_workbook TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS team_season_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    season_year INTEGER NOT NULL,
    feature_version TEXT NOT NULL,
    actual_win_pct REAL NOT NULL,
    sample_pa REAL NOT NULL,
    sample_ip REAL NOT NULL,
    sample_defend_cnt REAL NOT NULL,
    bat_obp REAL NOT NULL,
    bat_hr_per_pa REAL NOT NULL,
    bat_so_per_pa REAL NOT NULL,
    bat_wrc_plus REAL NOT NULL,
    pit_era REAL NOT NULL,
    pit_k9 REAL NOT NULL,
    pit_fip REAL NOT NULL,
    def_err_per_chance REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (team_id, season_year, feature_version),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS team_record_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    season_year INTEGER NOT NULL,
    feature_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    cv_alpha REAL NOT NULL,
    loocv_mse REAL NOT NULL,
    train_r2 REAL NOT NULL,
    actual_win_pct REAL NOT NULL,
    predicted_win_pct REAL NOT NULL,
    residual REAL NOT NULL,
    coefficients_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (team_id, season_year, feature_version, model_name),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS season_batting_all (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year INTEGER NOT NULL,
    team_code TEXT NOT NULL,
    team_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    acnt TEXT,
    plate_appearances REAL,
    obp REAL,
    home_run_cnt REAL,
    strike_out_cnt REAL,
    ops_plus REAL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS season_pitching_all (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year INTEGER NOT NULL,
    team_code TEXT NOT NULL,
    team_name TEXT NOT NULL,
    player_name TEXT NOT NULL,
    acnt TEXT,
    inning_pitched REAL,
    era REAL,
    k9 REAL,
    fip REAL,
    raw_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS team_game_results (
    game_result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    game_sno INTEGER NOT NULL,
    game_date TEXT,
    game_status INTEGER,
    game_status_text TEXT,
    visiting_team_code TEXT NOT NULL,
    home_team_code TEXT NOT NULL,
    visiting_score REAL,
    home_score REAL,
    raw_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (season_year, kind_code, game_sno)
);

CREATE TABLE IF NOT EXISTS prediction_tracking (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    season_year INTEGER NOT NULL,
    kind_code TEXT NOT NULL,
    game_sno INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    home_team_code TEXT NOT NULL,
    visiting_team_code TEXT NOT NULL,
    home_team_name TEXT NOT NULL,
    visiting_team_name TEXT NOT NULL,
    home_sp_name TEXT,
    visiting_sp_name TEXT,
    home_sp_acnt TEXT,
    visiting_sp_acnt TEXT,
    prob_home_win REAL NOT NULL,
    predicted_side TEXT NOT NULL,
    predicted_team_code TEXT NOT NULL,
    predicted_team_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    confidence_level TEXT NOT NULL,
    is_high_confidence INTEGER NOT NULL,
    threshold REAL NOT NULL,
    model_used TEXT NOT NULL,
    sp_available INTEGER NOT NULL,
    early_season INTEGER NOT NULL,
    actual_side TEXT NOT NULL,
    actual_team_code TEXT NOT NULL,
    actual_team_name TEXT NOT NULL,
    home_score REAL NOT NULL,
    visiting_score REAL NOT NULL,
    is_correct INTEGER NOT NULL,
    verified_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (season_year, kind_code, game_sno)
);

CREATE INDEX IF NOT EXISTS idx_players_team ON players(current_team_id);
CREATE INDEX IF NOT EXISTS idx_batting_team_year ON player_batting_season_stats(team_id, season_year, kind_code);
CREATE INDEX IF NOT EXISTS idx_pitching_team_year ON player_pitching_season_stats(team_id, season_year, kind_code);
CREATE INDEX IF NOT EXISTS idx_fielding_team_year ON player_fielding_season_stats(team_id, season_year, kind_code);
CREATE INDEX IF NOT EXISTS idx_team_records_team_year ON team_season_records(team_id, season_year, kind_code);
CREATE INDEX IF NOT EXISTS idx_team_features_team_year ON team_season_features(team_id, season_year, feature_version);
CREATE INDEX IF NOT EXISTS idx_team_predictions_team_year ON team_record_predictions(team_id, season_year, feature_version, model_name);
CREATE INDEX IF NOT EXISTS idx_season_batting_year ON season_batting_all(season_year, team_code);
CREATE INDEX IF NOT EXISTS idx_season_pitching_year ON season_pitching_all(season_year, team_code);
CREATE INDEX IF NOT EXISTS idx_team_game_results_year ON team_game_results(season_year, kind_code, game_status);
CREATE INDEX IF NOT EXISTS idx_prediction_tracking_date_conf ON prediction_tracking(game_date, is_high_confidence, confidence);
CREATE INDEX IF NOT EXISTS idx_prediction_tracking_model ON prediction_tracking(model_used, early_season, sp_available);
