"""Clean the FootLens injury file and attach the decision metrics.

Each row of the source file is one injury episode, not one player and not one
match. Around that episode the file stores up to three matches before the
injury, up to three matches missed, and up to three matches after return.
Match 1 is the match closest to the injury; match 3 is the furthest.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "player_injuries_impact.csv"

# Longer or more specific keys must be checked before shorter ones.
# "virus" is listed after "coronavirus" so a COVID row is not split later.
FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("hamstring", "Hamstring"),
    ("cruciate", "Knee"),
    ("meniscus", "Knee"),
    ("knee", "Knee"),
    ("ankle", "Ankle"),
    ("calf", "Calf"),
    ("adductor", "Groin"),
    ("groin", "Groin"),
    ("pubic", "Groin"),
    ("achilles", "Achilles"),
    ("hip", "Hip"),
    ("thigh", "Thigh"),
    ("metatarsal", "Foot"),
    ("foot", "Foot"),
    ("toe", "Foot"),
    ("shin", "Foot"),
    ("shoulder", "Shoulder"),
    ("concussion", "Head"),
    ("cheekbone", "Head"),
    ("facial", "Head"),
    ("head", "Head"),
    ("nose", "Head"),
    ("back", "Back"),
    ("dead leg", "Knock / bruise"),
    ("knock", "Knock / bruise"),
    ("bruise", "Knock / bruise"),
    ("coronavirus", "Illness"),
    ("quarantine", "Illness"),
    ("bronchitis", "Illness"),
    ("tonsillitis", "Illness"),
    ("shingles", "Illness"),
    ("malaria", "Illness"),
    ("infection", "Illness"),
    ("depression", "Illness"),
    ("fever", "Illness"),
    ("virus", "Illness"),
    ("cold", "Illness"),
    ("ill", "Illness"),
    ("muscular", "Muscle"),
    ("muscle", "Muscle"),
    ("unknown", "Unknown"),
)

FAMILY_ORDER = [
    "Hamstring",
    "Knee",
    "Ankle",
    "Calf",
    "Groin",
    "Thigh",
    "Muscle",
    "Hip",
    "Achilles",
    "Foot",
    "Back",
    "Shoulder",
    "Head",
    "Knock / bruise",
    "Illness",
    "Multiple",
    "Unknown",
    "Other",
]

POSITION_GROUPS = {
    "Goalkeeper": "Goalkeeper",
    "Center Back": "Defence",
    "Left Back": "Defence",
    "Right Back": "Defence",
    "Central Midfielder": "Midfield",
    "Defensive Midfielder": "Midfield",
    "Attacking Midfielder": "Midfield",
    "Left Midfielder": "Midfield",
    "Right Midfielder": "Midfield",
    "Center Forward": "Attack",
    "Left Winger": "Attack",
    "Right Winger": "Attack",
}

MISSING_TOKENS = {"", "n.a.", "na", "nan", "none"}
PHASES = ("before", "missed", "after")


def load_episodes(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Return one cleaned row per injury episode, plus a cleaning log.

    Rating averages and the goal-difference drop are added later by
    ``with_metrics``, because the dashboard can choose whether substitute
    appearances count.
    """
    raw = pd.read_csv(path)
    raw.columns = [column.strip() for column in raw.columns]
    duplicate_rows = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().reset_index(drop=True)

    stats = {
        "rows_in_file": int(len(raw) + duplicate_rows),
        "exact_duplicates_removed": duplicate_rows,
        "rating_typos_fixed": 0,
        "zero_ratings_dropped": 0,
        "invalid_ratings_dropped": 0,
        "substitute_ratings": 0,
    }

    frame = pd.DataFrame(
        {
            "player": raw["Name"].str.strip(),
            "club": raw["Team Name"].str.strip(),
            "position": _clean_position(raw["Position"]),
            "age": pd.to_numeric(raw["Age"], errors="coerce").astype("Int64"),
            "season": raw["Season"].str.strip(),
            "fifa_rating": pd.to_numeric(raw["FIFA rating"], errors="coerce"),
        }
    )
    frame["position_group"] = frame["position"].map(POSITION_GROUPS).fillna("Other")

    injury_key = raw["Injury"].map(_normalise_injury_key)
    frame["injury"] = injury_key.map(_label_from_key)
    frame["injury_family"] = injury_key.map(injury_family)

    injury_dates, ongoing_from_injury = _parse_dates(raw["Date of Injury"])
    return_dates, ongoing_from_return = _parse_dates(raw["Date of return"])
    frame["injury_date"] = injury_dates
    frame["return_date"] = return_dates
    frame["ongoing"] = ongoing_from_return.fillna(False).astype(bool)
    frame["injury_month"] = frame["injury_date"].dt.month.astype("Int64")

    duration = (frame["return_date"] - frame["injury_date"]).dt.days
    negative = duration.lt(0).fillna(False)
    too_long = duration.gt(800).fillna(False)
    frame["days_out"] = duration.mask(negative | too_long).astype("Int64")

    stats["injury_dates_unusable"] = int(frame["injury_date"].isna().sum())
    stats["return_dates_unusable"] = int(
        (frame["return_date"].isna() & ~frame["ongoing"]).sum()
    )
    stats["ongoing_absences"] = int(frame["ongoing"].sum())
    stats["negative_durations"] = int(negative.sum())
    stats["rows"] = int(len(frame))

    for index in (1, 2, 3):
        for phase, source in (
            ("before", "before_injury"),
            ("missed", "missed_match"),
            ("after", "after_injury"),
        ):
            prefix = f"m{index}_{phase}"
            stem = f"Match{index}_{source}"
            frame[f"{prefix}_result"] = raw[f"{stem}_Result"].map(_clean_token)
            frame[f"{prefix}_opponent"] = raw[f"{stem}_Opposition"].map(_clean_token)
            frame[f"{prefix}_gd"] = raw[f"{stem}_GD"].map(_parse_gd)
            if phase == "missed":
                continue
            parsed = raw[f"{stem}_Player_rating"].map(parse_rating)
            frame[f"{prefix}_rating"] = [item[0] for item in parsed]
            frame[f"{prefix}_sub"] = [item[1] for item in parsed]
            for issue in (item[2] for item in parsed):
                if issue == "typo":
                    stats["rating_typos_fixed"] += 1
                elif issue == "zero":
                    stats["zero_ratings_dropped"] += 1
                elif issue == "invalid":
                    stats["invalid_ratings_dropped"] += 1
            stats["substitute_ratings"] += int(sum(item[1] for item in parsed))

    frame = _add_repeat_flag(frame)
    return frame, stats


def with_metrics(frame: pd.DataFrame, include_subs: bool) -> pd.DataFrame:
    """Attach the two measures the rest of the dashboard is built on.

    Performance drop index
        Mean goal difference in the matches before the injury, minus the mean
        goal difference in the matches missed. Positive means the team’s goal
        difference was worse while the player was absent.

    Rating change
        Mean player rating after return, minus the mean rating before the
        injury. Positive means the player rated higher after coming back.
        Substitute ratings are left out unless ``include_subs`` is set.
    """
    out = frame.copy()
    out["gd_before"] = _window_mean(out, "before", "gd", include_subs=True)
    out["gd_missed"] = _window_mean(out, "missed", "gd", include_subs=True)
    out["gd_after"] = _window_mean(out, "after", "gd", include_subs=True)
    out["gd_drop"] = out["gd_before"] - out["gd_missed"]

    out["rating_before"] = _window_mean(out, "before", "rating", include_subs)
    out["rating_after"] = _window_mean(out, "after", "rating", include_subs)
    out["rating_change"] = out["rating_after"] - out["rating_before"]
    return out


def parse_rating(value) -> tuple[float, bool, str]:
    """Parse a match rating that may be missing, a substitute, or a typo.

    The file marks substitutes as ``6.4(S)`` and contains one doubled-dot
    typo (``5..8``). A rating of 0 is treated as missing: these are match
    ratings on a 1–10 scale, and 0 does not describe a performance.
    """
    if pd.isna(value):
        return np.nan, False, "missing"
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return np.nan, False, "missing"

    is_sub = bool(re.search(r"\(s\)", text, flags=re.IGNORECASE))
    core = re.sub(r"\(s\)", "", text, flags=re.IGNORECASE).strip()
    issue = ""
    if ".." in core:
        core = core.replace("..", ".")
        issue = "typo"
    try:
        number = float(core)
    except ValueError:
        return np.nan, is_sub, "invalid"
    if number == 0:
        return np.nan, is_sub, "zero"
    if number < 0 or number > 10:
        return np.nan, is_sub, "invalid"
    return number, is_sub, issue


def injury_family(key: str) -> str:
    """Group a cleaned injury label into a family a coach can act on."""
    if re.search(r"\band\b|,|&", key):
        return "Multiple"
    for token, family in FAMILY_RULES:
        if re.search(rf"\b{re.escape(token)}\b", key):
            return family
    return "Other"


def _normalise_injury_key(value: str) -> str:
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    key = key.replace("archilles", "achilles")
    if key == "ack injury":
        return "back injury"
    if key == "unknown injur":
        return "unknown injury"
    return key


def _label_from_key(key: str) -> str:
    return key.title()


def _clean_position(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    return cleaned.str.replace("winger", "Winger", regex=False)


def _clean_token(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return None
    return text


def _parse_gd(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.lower() in MISSING_TOKENS:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_dates(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse mixed date text.

    Many rows omit the space after the comma (``Dec 6,2019``). Three returns
    say ``Present`` and stay open. A year outside 2015–2026, such as the
    typed ``0202``, is discarded rather than repaired.
    """
    text = series.fillna("").astype(str).str.strip()
    ongoing = text.str.lower().eq("present")
    invalid = text.str.lower().isin(MISSING_TOKENS | {"present"})
    normalised = text.str.replace(r",(?=\S)", ", ", regex=True)
    parsed = pd.to_datetime(normalised.mask(invalid), errors="coerce", format="mixed")
    bad_year = parsed.dt.year.lt(2015) | parsed.dt.year.gt(2026)
    parsed = parsed.mask(bad_year.fillna(False))
    return parsed, ongoing


def _window_mean(
    frame: pd.DataFrame, phase: str, kind: str, include_subs: bool
) -> pd.Series:
    columns = []
    for index in (1, 2, 3):
        values = frame[f"m{index}_{phase}_{kind}"]
        if kind == "rating" and not include_subs:
            values = values.mask(frame[f"m{index}_{phase}_sub"])
        columns.append(values)
    return pd.concat(columns, axis=1).mean(axis=1, skipna=True)


def _add_repeat_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark a later episode for the same player.

    Only episodes with a usable injury date take part in the order. An
    undated episode is not called a recurrence.
    """
    out = frame.copy()
    dated = out["injury_date"].notna()
    ordered = out.loc[dated].sort_values(["player", "injury_date"])
    prior = ordered.groupby("player", sort=False).cumcount()
    out["prior_injuries"] = prior.reindex(out.index).astype("Int64")
    out["is_repeat"] = prior.reindex(out.index).fillna(0).gt(0) & dated
    return out
