import argparse
import sqlite3
import time

import requests


SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
BOXSCORE_URL = "https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
STANDINGS_URL = "https://statsapi.mlb.com/api/v1/standings"
DB_PATH = "mlb.sqlite"
GAME_TYPE = "R"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS team_game_results (
    season_year INTEGER,
    game_pk     INTEGER PRIMARY KEY,
    game_date   TEXT,
    home_team_id   INTEGER,
    home_team_name TEXT,
    vis_team_id    INTEGER,
    vis_team_name  TEXT,
    home_score  INTEGER,
    vis_score   INTEGER,
    game_type   TEXT,
    status      TEXT,
    winner      TEXT
);

CREATE TABLE IF NOT EXISTS game_starting_pitchers (
    season_year  INTEGER,
    game_pk      INTEGER,
    game_date    TEXT,
    team_id      INTEGER,
    team_name    TEXT,
    pitcher_name TEXT,
    pitcher_id   INTEGER,
    ip           REAL,
    er           INTEGER,
    k            INTEGER,
    bb           INTEGER,
    h            INTEGER,
    PRIMARY KEY (game_pk, team_id)
);

CREATE TABLE IF NOT EXISTS team_season_records (
    season_year INTEGER,
    team_id     INTEGER,
    team_name   TEXT,
    wins        INTEGER,
    losses      INTEGER,
    win_pct     REAL,
    division    TEXT,
    league      TEXT,
    PRIMARY KEY (season_year, team_id)
);
"""


def build_parser():
    parser = argparse.ArgumentParser(
        description="Scrape MLB regular-season results and starting pitchers into SQLite."
    )
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--refresh-range",
        action="store_true",
        help="Re-fetch and upsert the selected year range without deleting old rows.",
    )
    return parser


def connect_db():
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.executescript(SCHEMA_SQL)
    return connection


def get_existing_game_pks(connection, start_year, end_year):
    rows = connection.execute(
        """
        SELECT game_pk
        FROM team_game_results
        WHERE season_year BETWEEN ? AND ?
        """,
        (start_year, end_year),
    ).fetchall()
    return {row[0] for row in rows}


def month_end_day(year, month):
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    if (year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
        return 29
    return 28


def sleep_between_calls(state):
    state["call_count"] += 1
    time.sleep(0.3 + ((state["call_count"] - 1) % 3) * 0.1)


def api_get_json(session, url, state, params=None):
    try:
        response = session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        print(f"warning: request failed url={url} params={params} error={exc}")
        return None
    finally:
        sleep_between_calls(state)


def to_int(value):
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None


def to_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def normalize_status(game):
    status = (((game or {}).get("status") or {}).get("detailedState")) or ""
    if status.lower() == "final":
        return "Final"
    return status


def get_game_scores(game):
    teams = (game or {}).get("teams") or {}
    home_score = to_int(((teams.get("home") or {}).get("score")))
    vis_score = to_int(((teams.get("away") or {}).get("score")))

    if home_score is None or vis_score is None:
        linescore = game.get("linescore") or {}
        teams_linescore = linescore.get("teams") or {}
        if home_score is None:
            home_score = to_int(((teams_linescore.get("home") or {}).get("runs")))
        if vis_score is None:
            vis_score = to_int(((teams_linescore.get("away") or {}).get("runs")))
    return home_score, vis_score


def get_winner(home_score, vis_score):
    if home_score is None or vis_score is None:
        return None
    if home_score > vis_score:
        return "home"
    if vis_score > home_score:
        return "vis"
    return None


def upsert_game_result(connection, row):
    connection.execute(
        """
        INSERT INTO team_game_results(
            season_year,
            game_pk,
            game_date,
            home_team_id,
            home_team_name,
            vis_team_id,
            vis_team_name,
            home_score,
            vis_score,
            game_type,
            status,
            winner
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_pk) DO UPDATE SET
            season_year=excluded.season_year,
            game_date=excluded.game_date,
            home_team_id=excluded.home_team_id,
            home_team_name=excluded.home_team_name,
            vis_team_id=excluded.vis_team_id,
            vis_team_name=excluded.vis_team_name,
            home_score=excluded.home_score,
            vis_score=excluded.vis_score,
            game_type=excluded.game_type,
            status=excluded.status,
            winner=excluded.winner
        """,
        row,
    )


def upsert_starting_pitcher(connection, row):
    connection.execute(
        """
        INSERT INTO game_starting_pitchers(
            season_year,
            game_pk,
            game_date,
            team_id,
            team_name,
            pitcher_name,
            pitcher_id,
            ip,
            er,
            k,
            bb,
            h
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_pk, team_id) DO UPDATE SET
            season_year=excluded.season_year,
            game_date=excluded.game_date,
            team_name=excluded.team_name,
            pitcher_name=excluded.pitcher_name,
            pitcher_id=excluded.pitcher_id,
            ip=excluded.ip,
            er=excluded.er,
            k=excluded.k,
            bb=excluded.bb,
            h=excluded.h
        """,
        row,
    )


def upsert_team_record(connection, row):
    connection.execute(
        """
        INSERT INTO team_season_records(
            season_year,
            team_id,
            team_name,
            wins,
            losses,
            win_pct,
            division,
            league
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(season_year, team_id) DO UPDATE SET
            team_name=excluded.team_name,
            wins=excluded.wins,
            losses=excluded.losses,
            win_pct=excluded.win_pct,
            division=excluded.division,
            league=excluded.league
        """,
        row,
    )


def fetch_team_map(session, season_year, state):
    payload = api_get_json(
        session,
        TEAMS_URL,
        state,
        params={"sportId": 1, "season": season_year},
    )
    team_map = {}
    for team in (payload or {}).get("teams") or []:
        team_id = to_int(team.get("id"))
        team_name = team.get("name")
        if team_id is not None and team_name:
            team_map[team_id] = team_name
    return team_map


def extract_sp_rows(season_year, game_pk, game_date, game_row, boxscore_payload):
    rows = []
    teams_payload = (boxscore_payload or {}).get("teams") or {}
    for side, team_id_index, team_name_index in (
        ("home", 3, 4),
        ("away", 5, 6),
    ):
        team_payload = teams_payload.get(side) or {}
        pitcher_ids = team_payload.get("pitchers") or []
        players = team_payload.get("players") or {}
        if not pitcher_ids:
            print(f"warning: no pitcher list game_pk={game_pk} side={side}")
            continue

        pitcher_id = to_int(pitcher_ids[0])
        if pitcher_id is None:
            print(f"warning: invalid starter id game_pk={game_pk} side={side}")
            continue

        player_key = f"ID{pitcher_id}"
        player_payload = players.get(player_key) or {}
        person = player_payload.get("person") or {}
        pitching_stats = ((player_payload.get("stats") or {}).get("pitching")) or {}
        pitcher_name = person.get("fullName")

        if not pitcher_name:
            print(f"warning: missing starter name game_pk={game_pk} side={side} pitcher_id={pitcher_id}")
            continue

        rows.append(
            (
                season_year,
                game_pk,
                game_date,
                game_row[team_id_index],
                game_row[team_name_index],
                pitcher_name,
                pitcher_id,
                to_float(pitching_stats.get("inningsPitched")),
                to_int(pitching_stats.get("earnedRuns")),
                to_int(pitching_stats.get("strikeOuts")),
                to_int(pitching_stats.get("baseOnBalls")),
                to_int(pitching_stats.get("hits")),
            )
        )
    return rows


def process_schedule_game(connection, session, season_year, team_map, game, state):
    game_pk = to_int(game.get("gamePk"))
    official_date = game.get("officialDate")
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_team = home.get("team") or {}
    away_team = away.get("team") or {}

    home_team_id = to_int(home_team.get("id"))
    vis_team_id = to_int(away_team.get("id"))
    home_team_name = home_team.get("name") or team_map.get(home_team_id)
    vis_team_name = away_team.get("name") or team_map.get(vis_team_id)
    home_score, vis_score = get_game_scores(game)

    if game_pk is None or not official_date or home_team_id is None or vis_team_id is None:
        print(f"warning: malformed schedule row season={season_year} game={game}")
        return 0

    game_row = (
        season_year,
        game_pk,
        official_date,
        home_team_id,
        home_team_name,
        vis_team_id,
        vis_team_name,
        home_score,
        vis_score,
        game.get("gameType") or GAME_TYPE,
        "Final",
        get_winner(home_score, vis_score),
    )
    upsert_game_result(connection, game_row)

    boxscore_payload = api_get_json(
        session,
        BOXSCORE_URL.format(game_pk=game_pk),
        state,
    )
    if not boxscore_payload:
        return 0

    sp_rows = extract_sp_rows(season_year, game_pk, official_date, game_row, boxscore_payload)
    if not sp_rows:
        return 0

    for sp_row in sp_rows:
        upsert_starting_pitcher(connection, sp_row)
    return len(sp_rows)


def scrape_schedule_for_year(connection, session, season_year, team_map, existing_game_pks, refresh_range, state):
    games_saved = 0
    sp_saved = 0

    for month in range(1, 13):
        start_date = f"{season_year}-{month:02d}-01"
        end_date = f"{season_year}-{month:02d}-{month_end_day(season_year, month):02d}"
        payload = api_get_json(
            session,
            SCHEDULE_URL,
            state,
            params={
                "sportId": 1,
                "startDate": start_date,
                "endDate": end_date,
                "gameType": GAME_TYPE,
                "hydrate": "linescore",
            },
        )
        if not payload:
            continue

        for date_block in payload.get("dates") or []:
            for game in date_block.get("games") or []:
                status = normalize_status(game)
                if status != "Final":
                    continue

                game_pk = to_int(game.get("gamePk"))
                if game_pk is None:
                    print(f"warning: missing gamePk season={season_year} month={month}")
                    continue

                if not refresh_range and game_pk in existing_game_pks:
                    continue

                try:
                    sp_count = process_schedule_game(
                        connection=connection,
                        session=session,
                        season_year=season_year,
                        team_map=team_map,
                        game=game,
                        state=state,
                    )
                    games_saved += 1
                    sp_saved += sp_count
                    existing_game_pks.add(game_pk)
                except Exception as exc:
                    print(f"warning: failed to process game_pk={game_pk} error={exc}")

        connection.commit()

    return games_saved, sp_saved


def scrape_standings_for_year(connection, session, season_year, team_map, state):
    payload = api_get_json(
        session,
        STANDINGS_URL,
        state,
        params={
            "leagueId": "103,104",
            "season": season_year,
            "standingsType": "regularSeason",
        },
    )
    if not payload:
        return

    for record_block in payload.get("records") or []:
        division = ((record_block.get("division") or {}).get("name")) or None
        league = ((record_block.get("league") or {}).get("name")) or None
        for team_record in record_block.get("teamRecords") or []:
            team = team_record.get("team") or {}
            team_id = to_int(team.get("id"))
            team_name = team.get("name") or team_map.get(team_id)
            if team_id is None or not team_name:
                print(f"warning: malformed standings row season={season_year} row={team_record}")
                continue

            upsert_team_record(
                connection,
                (
                    season_year,
                    team_id,
                    team_name,
                    to_int(team_record.get("wins")),
                    to_int(team_record.get("losses")),
                    to_float(team_record.get("winningPercentage")),
                    division,
                    league,
                ),
            )
    connection.commit()


def print_table_counts(connection):
    for table_name in (
        "team_game_results",
        "game_starting_pitchers",
        "team_season_records",
    ):
        count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"{table_name}: {count}")


def main():
    args = build_parser().parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    connection = connect_db()
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "MLB-Boxscore-Scraper/1.0",
            "Accept": "application/json",
        }
    )
    state = {"call_count": 0}

    try:
        existing_game_pks = get_existing_game_pks(connection, args.start_year, args.end_year)
        for season_year in range(args.start_year, args.end_year + 1):
            team_map = fetch_team_map(session, season_year, state)
            games_saved, sp_saved = scrape_schedule_for_year(
                connection=connection,
                session=session,
                season_year=season_year,
                team_map=team_map,
                existing_game_pks=existing_game_pks,
                refresh_range=args.refresh_range,
                state=state,
            )
            scrape_standings_for_year(
                connection=connection,
                session=session,
                season_year=season_year,
                team_map=team_map,
                state=state,
            )
            print(f"[{season_year}] games: {games_saved}, SPs: {sp_saved}")

        print_table_counts(connection)
        print(f"saved_db={DB_PATH}")
    finally:
        session.close()
        connection.close()


if __name__ == "__main__":
    main()
