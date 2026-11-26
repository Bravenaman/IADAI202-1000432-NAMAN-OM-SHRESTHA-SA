"""FootLens dashboard: player injuries and team performance.

The page is arranged around the decisions in the brief. Filters apply to
every figure. Chart text is computed from the filtered rows, so a change in
club or season changes the sentence under the chart as well as the marks.
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from preprocess import FAMILY_ORDER, load_episodes, with_metrics

LOGO_PATH = Path(__file__).parent / "assets" / "logo.png"

st.set_page_config(
    page_title="FootLens Analytics",
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROUP_COLORS = {
    "Goalkeeper": "#94a3b8",
    "Defence": "#38bdf8",
    "Midfield": "#fbbf24",
    "Attack": "#fb7185",
}
DROP_COLOR = "#fb7185"
GAIN_COLOR = "#34d399"
NEUTRAL = "#e5e7eb"
MUSCLE_FAMILIES = {
    "Hamstring",
    "Knee",
    "Ankle",
    "Calf",
    "Groin",
    "Thigh",
    "Muscle",
    "Hip",
    "Achilles",
}

# Same order as sorted club names in the Club chips.
CLUB_BADGES = (
    "arsenal",
    "aston-villa",
    "brentford",
    "burnley",
    "everton",
    "man-united",
    "newcastle",
    "tottenham",
)

TIMELINE = (
    ("Before 3", "m3_before", True),
    ("Before 2", "m2_before", True),
    ("Before 1", "m1_before", True),
    ("Missed 1", "m1_missed", False),
    ("Missed 2", "m2_missed", False),
    ("Missed 3", "m3_missed", False),
    ("After 1", "m1_after", True),
    ("After 2", "m2_after", True),
    ("After 3", "m3_after", True),
)


def _club_badge_css() -> str:
    """Badge then name, only on the Club chips. Other filters keep the plain check."""
    club = (
        'section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] '
        '[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(3) '
        'button[data-variant="pills"]'
    )
    rules = [
        f"{club} {{"
        "padding: 6px 12px 6px 66px !important;"
        "background-color: #ffffff !important;"
        "background-repeat: no-repeat !important;"
        "background-position: 32px center !important;"
        "background-size: 24px 24px !important;"
        "}"
    ]
    folder = Path(__file__).parent / "assets" / "badges"
    for index, name in enumerate(CLUB_BADGES, start=1):
        encoded = base64.b64encode((folder / f"{name}.png").read_bytes()).decode()
        rules.append(
            f'{club}:nth-child({index}) {{'
            f'background-image: url("data:image/png;base64,{encoded}") !important;'
            "}"
        )
    return "\n".join(rules)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 3rem; }
        div[data-testid="stElementContainer"]:has(.brand),
        div[data-testid="stMarkdown"]:has(.brand),
        div[data-testid="stMarkdown"]:has(.brand) > div,
        div[data-testid="stMarkdownContainer"]:has(.brand) {
            height: auto !important;
            min-height: 78px !important;
            overflow: visible !important;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 22px 0 0.15rem;
            overflow: visible;
            line-height: 1;
        }
        .brand img {
            width: 48px !important;
            height: 48px !important;
            max-width: none !important;
            max-height: none !important;
            object-fit: contain;
            flex: none;
            display: block;
            overflow: visible;
        }
        .brand h1 {
            margin: 0;
            padding: 0;
            font-size: 2.25rem;
            line-height: 1.1;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #f8fafc;
        }
        div[data-testid="stMarkdown"]:has(.brand) { margin-bottom: 0; }
        div[data-testid="stMetric"] {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            padding: 12px 14px;
        }
        [data-testid="stMain"] [data-testid="stCaptionContainer"],
        [data-testid="stMain"] [data-testid="stCaptionContainer"] p,
        [data-testid="stMain"] [data-testid="stCaptionContainer"] span {
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] {
            background: #f7f8fa !important;
            border: none !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            background: #f7f8fa;
            padding-top: 1.2rem;
        }
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #111827;
        }
        .filter-heading {
            font-size: 15px;
            font-weight: 700;
            letter-spacing: -0.01em;
            color: #111827 !important;
            margin: 0 0 10px;
            padding-top: 16px;
            border-top: 1px solid #e6e8ee;
        }
        .filter-heading.first {
            border-top: none;
            padding-top: 0;
        }
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] {
            margin: -2px 0 6px;
        }
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
            font-size: 13.5px !important;
            font-weight: 600 !important;
            color: #1f2937 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label > div:first-of-type {
            background: #ffffff !important;
            border: 1.5px solid #d0d5dd !important;
            border-radius: 4px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label[data-selected="true"] > div:first-of-type {
            background: #2f6fed !important;
            border-color: #2f6fed !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label[data-selected="true"] svg,
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label[data-selected="true"] polyline {
            color: #ffffff !important;
            stroke: #ffffff !important;
        }
        section[data-testid="stSidebar"] [data-testid="stButtonGroup"] {
            margin-bottom: 4px;
        }
        section[data-testid="stSidebar"] [data-testid="stButtonGroup"] > div {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
            width: 100% !important;
        }
        section[data-testid="stSidebar"] button[data-variant="pills"] {
            position: relative;
            background-color: #ffffff !important;
            color: #1f2937 !important;
            border: 1px solid #e4e7ec !important;
            border-radius: 999px !important;
            padding: 7px 12px 7px 32px !important;
            min-height: 34px !important;
            font-size: 13.5px !important;
            font-weight: 500 !important;
            line-height: 1.2 !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] button[data-variant="pills"]::before {
            content: "";
            position: absolute;
            left: 10px;
            top: 50%;
            width: 16px;
            height: 16px;
            transform: translateY(-50%);
            border-radius: 50%;
            border: 1.5px solid #d0d5dd;
            background: #ffffff;
            box-sizing: border-box;
        }
        section[data-testid="stSidebar"] button[data-variant="pills"][data-selected] {
            background-color: #ffffff !important;
            color: #111827 !important;
            border-color: #d7deea !important;
        }
        section[data-testid="stSidebar"] button[data-variant="pills"][data-selected]::before {
            content: "✓";
            border: none;
            background: #2f6fed;
            color: #ffffff;
            font-size: 11px;
            font-weight: 700;
            line-height: 16px;
            text-align: center;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="group"] > div > div:first-child {
            filter: grayscale(1) brightness(0.25);
            height: 4px !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="group"] > div > div:nth-child(2),
        section[data-testid="stSidebar"] [data-testid="stSlider"] [role="group"] > div > div:nth-child(3) {
            background: #111827 !important;
            width: 18px !important;
            height: 18px !important;
            border: 2px solid #ffffff !important;
            box-shadow: 0 0 0 1px #111827 !important;
        }
        .result-btn {
            display: inline-block;
            margin: 18px 0 8px;
            background: #2f6fed;
            color: #ffffff !important;
            border-radius: 12px;
            padding: 11px 18px;
            font-size: 15px;
            font-weight: 650;
            letter-spacing: -0.01em;
        }
        """
        + _club_badge_css()
        + """
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_data() -> tuple[pd.DataFrame, dict]:
    return load_episodes()


def style(fig: go.Figure, height: int = 440, margin_l: int = 64) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=margin_l, r=36, t=72, b=56),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e5e7eb", size=13),
        title=dict(font=dict(size=16, color="#f8fafc")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(color="#f8fafc")),
        hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_color="#f8fafc"),
    )
    fig.update_xaxes(
        gridcolor="#475569",
        gridwidth=1,
        zerolinecolor="#94a3b8",
        zerolinewidth=1.5,
        showline=True,
        linecolor="#64748b",
        linewidth=1,
        color="#e5e7eb",
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="#475569",
        gridwidth=1,
        zerolinecolor="#94a3b8",
        zerolinewidth=1.5,
        showline=True,
        linecolor="#64748b",
        linewidth=1,
        color="#e5e7eb",
        automargin=True,
        ticklabelstandoff=6,
    )
    return fig


def signed(value: float, digits: int = 2) -> str:
    return f"{value:+.{digits}f}"


def _shown(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    return str(value)


def most_common(series: pd.Series) -> str:
    series = series.dropna()
    if series.empty:
        return "—"
    return str(series.value_counts().index[0])


def _filter_heading(title: str, first: bool = False) -> None:
    css_class = "filter-heading first" if first else "filter-heading"
    st.sidebar.markdown(f'<p class="{css_class}">{title}</p>', unsafe_allow_html=True)


def _toggle_all(key: str, options: tuple) -> None:
    st.session_state[key] = list(options) if st.session_state[f"{key}_all"] else []


def _match_all(key: str, options: tuple) -> None:
    chosen = st.session_state.get(key) or []
    st.session_state[f"{key}_all"] = set(chosen) == set(options)


def _pills(label: str, options: list, key: str, selection_mode: str = "multi", default=None):
    """Chip row. Multi-select starts with every option on, and a Select all box stays in step."""
    if selection_mode == "multi":
        choices = tuple(options)
        all_key = f"{key}_all"
        if key not in st.session_state:
            st.session_state[key] = list(choices)
        if all_key not in st.session_state:
            st.session_state[all_key] = set(st.session_state[key] or []) == set(choices)
        st.sidebar.checkbox(
            "Select all",
            key=all_key,
            on_change=_toggle_all,
            args=(key, choices),
        )
        value = st.sidebar.pills(
            label,
            options,
            selection_mode="multi",
            key=key,
            label_visibility="collapsed",
            wrap=True,
            on_change=_match_all,
            args=(key, choices),
        )
        return list(value or [])

    value = st.sidebar.pills(
        label,
        options,
        selection_mode=selection_mode,
        default=default,
        key=key,
        label_visibility="collapsed",
        wrap=True,
    )
    return value


WALLPAPERS = {
    "Arsenal": "arsenal.jpg",
    "Aston Villa": "aston-villa.jpg",
    "Brentford": "brentford.jpg",
    "Burnley": "burnley.jpg",
    "Everton": "everton.jpg",
    "Man United": "man-united.jpg",
    "Newcastle": "newcastle.jpg",
    "Tottenham": "tottenham.jpg",
}


def club_wallpaper(clubs: list[str]) -> None:
    """Full-page club art only when a single club with a wallpaper is selected."""
    if len(clubs) != 1 or clubs[0] not in WALLPAPERS:
        return
    path = Path(__file__).parent / "assets" / "backgrounds" / WALLPAPERS[clubs[0]]
    encoded = base64.b64encode(path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stApp"],
        [data-testid="stAppViewContainer"] {{
            background-color: #0b1220 !important;
            background-image:
                linear-gradient(rgba(11, 18, 32, 0.42), rgba(11, 18, 32, 0.55)),
                url("data:image/jpeg;base64,{encoded}") !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background: transparent !important;
        }}
        [data-testid="stHeader"] {{
            background: rgba(11, 18, 32, 0.2) !important;
        }}
        section[data-testid="stSidebar"] {{
            background: #f7f8fa !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def filter_episodes(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool, list[str]]:
    _filter_heading("Club", first=True)
    clubs = _pills("Club", sorted(frame["club"].unique()), "club_pills")

    _filter_heading("Season")
    seasons = _pills("Season", sorted(frame["season"].unique()), "season_pills")

    _filter_heading("Position")
    positions = _pills("Position", sorted(frame["position"].unique()), "position_pills")

    _filter_heading("Injury family")
    families_present = [family for family in FAMILY_ORDER if family in set(frame["injury_family"])]
    families = _pills("Injury family", families_present, "family_pills")

    _filter_heading("Age")
    age_low = int(frame["age"].min())
    age_high = int(frame["age"].max())
    age_range = st.sidebar.slider(
        "Age",
        age_low,
        age_high,
        (age_low, age_high),
        key="age_range",
        label_visibility="collapsed",
    )

    _filter_heading("Ratings")
    rating_choice = _pills(
        "Ratings",
        ["Full appearances", "Count substitutes"],
        "rating_pills",
        selection_mode="single",
        default="Full appearances",
    )
    include_subs = rating_choice == "Count substitutes"

    missing = [
        name
        for name, chosen in (
            ("club", clubs),
            ("season", seasons),
            ("position", positions),
            ("injury family", families),
        )
        if not chosen
    ]
    if missing:
        st.sidebar.warning("Select at least one " + ", ".join(missing) + ".")

    mask = (
        frame["club"].isin(clubs)
        & frame["season"].isin(seasons)
        & frame["position"].isin(positions)
        & frame["injury_family"].isin(families)
        & frame["age"].between(age_range[0], age_range[1])
    )
    filtered = with_metrics(frame.loc[mask].copy(), include_subs)
    st.sidebar.markdown(
        f'<div class="result-btn">Show {len(filtered)} episodes</div>',
        unsafe_allow_html=True,
    )
    return filtered, include_subs, list(clubs)


def injury_drop_chart(frame: pd.DataFrame, minimum: int) -> None:
    grouped = (
        frame.dropna(subset=["gd_drop"])
        .groupby("injury", as_index=False)
        .agg(drop=("gd_drop", "mean"), episodes=("gd_drop", "size"), median_drop=("gd_drop", "median"))
    )
    qualified = grouped[grouped["episodes"] >= minimum].sort_values("drop", ascending=False)
    st.subheader("Injuries with the largest team drop")
    st.caption(
        "Performance drop index: mean goal difference before the injury, minus mean goal difference "
        "in the matches missed. Positive means the team’s goal difference was worse during the absence."
    )
    if qualified.empty:
        st.info(f"No injury type in this filter has at least {minimum} episodes with a before-and-missed goal difference.")
        return

    top = qualified.head(10).sort_values("drop", ascending=True)
    colors = [DROP_COLOR if value > 0 else GAIN_COLOR for value in top["drop"]]
    fig = go.Figure(
        go.Bar(
            x=top["drop"],
            y=top["injury"],
            orientation="h",
            marker_color=colors,
            customdata=np.column_stack([top["episodes"], top["median_drop"]]),
            hovertemplate=(
                "%{y}<br>Mean drop %{x:.2f}<br>Median %{customdata[1]:.2f}"
                "<br>%{customdata[0]} episodes<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1)
    style(fig, height=520, margin_l=168)
    fig.update_layout(title="Top injuries by mean goal-difference drop", xaxis_title="Mean drop index", yaxis_title="")
    st.plotly_chart(fig, theme=None)

    leader = qualified.iloc[0]
    if leader["drop"] > 0:
        sentence = (
            f"{leader['injury']} has the largest average drop in this filter "
            f"({signed(leader['drop'])} across {int(leader['episodes'])} episodes)."
        )
    else:
        sentence = (
            f"No injury type here averages a worse goal difference during the absence. "
            f"The highest value is {leader['injury']} at {signed(leader['drop'])}."
        )
    st.caption(sentence)


def burden_chart(frame: pd.DataFrame) -> None:
    st.subheader("Days lost by injury family")
    st.caption("Total days from injury date to return. Open absences and unusable dates are excluded.")
    usable = frame.dropna(subset=["days_out"])
    if usable.empty:
        st.info("None of the filtered episodes have a usable duration.")
        return
    totals = (
        usable.groupby("injury_family", as_index=False)["days_out"]
        .agg(days="sum", episodes="size", median_days="median")
        .sort_values("days", ascending=True)
    )
    fig = go.Figure(
        go.Bar(
            x=totals["days"],
            y=totals["injury_family"],
            orientation="h",
            marker_color="#38bdf8",
            customdata=np.column_stack([totals["episodes"], totals["median_days"]]),
            hovertemplate=(
                "%{y}<br>%{x:.0f} days lost<br>Median %{customdata[1]:.0f} days"
                "<br>%{customdata[0]} episodes<extra></extra>"
            ),
        )
    )
    style(fig, height=520, margin_l=140)
    fig.update_layout(title="Injury burden, total days", xaxis_title="Days lost", yaxis_title="")
    st.plotly_chart(fig, theme=None)
    heaviest = totals.iloc[-1]
    st.caption(
        f"{heaviest['injury_family']} accounts for {heaviest['days']:.0f} days "
        f"across {int(heaviest['episodes'])} episodes in this filter "
        f"(median {heaviest['median_days']:.0f} days)."
    )


def age_scatter(frame: pd.DataFrame) -> None:
    st.subheader("Age and the goal-difference drop")
    st.caption("Each point is one injury episode. The line at zero is a team goal difference that did not move.")
    usable = frame.dropna(subset=["gd_drop", "age"]).copy()
    if usable.empty:
        st.info("No episodes in this filter have both an age and a goal-difference drop.")
        return
    usable["age_num"] = usable["age"].astype(int)
    fig = go.Figure()
    for group, color in GROUP_COLORS.items():
        part = usable[usable["position_group"] == group]
        if part.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=part["age_num"],
                y=part["gd_drop"],
                mode="markers",
                name=group,
                marker=dict(
                    color=color,
                    size=np.clip((part["fifa_rating"] - 64) / 2.2, 8, 18),
                    opacity=1,
                    line=dict(width=1, color="#0f172a"),
                ),
                customdata=np.column_stack(
                    [part["player"], part["club"], part["injury"], part["season"], part["fifa_rating"]]
                ),
                hovertemplate=(
                    "%{customdata[0]} · %{customdata[1]}<br>%{customdata[2]} · %{customdata[3]}"
                    "<br>Age %{x}<br>Drop %{y:.2f}<br>FIFA %{customdata[4]:.0f}<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_color="#cbd5e1", line_width=1)
    style(fig, height=480)
    fig.update_layout(
        title="Age against performance drop index",
        xaxis_title="Age",
        yaxis_title="Goal-difference drop",
    )
    st.plotly_chart(fig, theme=None)
    pair = usable[["age_num", "gd_drop"]]
    if len(pair) >= 3:
        correlation = float(pair.corr().iloc[0, 1])
        st.caption(
            f"In this filter the correlation between age and the drop index is {correlation:+.2f}. "
            "Marker size follows the FIFA rating."
        )


def heatmap(frame: pd.DataFrame) -> None:
    st.subheader("When injuries cluster")
    st.caption("Episode counts by club and month of the injury date. Episodes with an unusable date are omitted.")
    usable = frame.dropna(subset=["injury_month"]).copy()
    if usable.empty:
        st.info("No filtered episodes have a usable injury month.")
        return
    usable["injury_month"] = usable["injury_month"].astype(int)
    counts = pd.crosstab(usable["club"], usable["injury_month"])
    counts = counts.reindex(columns=range(1, 13), fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    counts.columns = MONTHS
    fig = go.Figure(
        go.Heatmap(
            z=counts.values,
            x=list(counts.columns),
            y=list(counts.index),
            colorscale="YlOrRd",
            hovertemplate="%{y} · %{x}<br>%{z} episodes<extra></extra>",
            colorbar=dict(
                title=dict(text="Episodes", font=dict(color="#f8fafc")),
                tickfont=dict(color="#e5e7eb"),
            ),
        )
    )
    style(fig, height=420, margin_l=110)
    fig.update_layout(title="Injury frequency by month and club")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, theme=None)
    stacked = counts.stack()
    club, month = stacked.idxmax()
    st.caption(f"The busiest cell in this filter is {club} in {month} ({int(stacked.max())} episodes).")


def club_drop_chart(frame: pd.DataFrame) -> None:
    st.subheader("Which clubs drop further")
    st.caption("Mean goal-difference drop while the injured player is out. Compared within the filtered seasons only.")
    usable = frame.dropna(subset=["gd_drop"])
    if usable.empty:
        st.info("No filtered episodes have a goal-difference drop.")
        return
    grouped = (
        usable.groupby("club", as_index=False)
        .agg(drop=("gd_drop", "mean"), episodes=("gd_drop", "size"))
        .sort_values("drop", ascending=True)
    )
    colors = [DROP_COLOR if value > 0 else GAIN_COLOR for value in grouped["drop"]]
    fig = go.Figure(
        go.Bar(
            x=grouped["drop"],
            y=grouped["club"],
            orientation="h",
            marker_color=colors,
            customdata=grouped["episodes"],
            hovertemplate="%{y}<br>Mean drop %{x:.2f}<br>%{customdata} episodes<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1)
    style(fig, height=420, margin_l=110)
    fig.update_layout(title="Mean goal-difference drop by club", xaxis_title="Mean drop index", yaxis_title="")
    st.plotly_chart(fig, theme=None)


def timeline(frame: pd.DataFrame, include_subs: bool) -> None:
    st.subheader("Player timeline around one injury")
    st.caption(
        "Match 1 is the match closest to the injury. The shaded band is the absence. "
        "Bars are the team goal difference. The line is the player’s match rating."
    )
    if frame.empty:
        st.info("No episodes match the current filters.")
        return
    players = sorted(frame["player"].unique())
    player = st.selectbox("Player", players, index=0)
    episodes = frame[frame["player"] == player].sort_values("injury_date", na_position="last")
    labels = []
    for row in episodes.itertuples():
        when = row.injury_date.strftime("%d %b %Y") if pd.notna(row.injury_date) else "date unknown"
        labels.append(f"{row.injury} · {when} · {row.season}")
    choice = st.selectbox("Injury episode", labels, index=0)
    row = episodes.iloc[labels.index(choice)]

    labels_x = []
    ratings = []
    symbols = []
    goal_diff = []
    hover = []
    for label, prefix, has_rating in TIMELINE:
        labels_x.append(label)
        gd = row.get(f"{prefix}_gd", np.nan)
        goal_diff.append(gd)
        result = _shown(row.get(f"{prefix}_result"))
        opponent = _shown(row.get(f"{prefix}_opponent"))
        rating = np.nan
        symbol = "circle"
        if has_rating:
            rating = row.get(f"{prefix}_rating", np.nan)
            is_sub = bool(row.get(f"{prefix}_sub", False))
            if is_sub and not include_subs:
                rating = np.nan
            if is_sub and include_subs and pd.notna(rating):
                symbol = "circle-open"
        ratings.append(rating)
        symbols.append(symbol)
        detail = f"{label}<br>{result} vs {opponent}"
        if pd.notna(gd):
            detail += f"<br>Goal difference {gd:+.0f}"
        if pd.notna(rating):
            detail += f"<br>Rating {rating:.1f}"
        hover.append(detail)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels_x,
            y=goal_diff,
            name="Goal difference",
            marker_color="#38bdf8",
            marker_line=dict(width=0),
            hoverinfo="skip",
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels_x,
            y=ratings,
            name="Player rating",
            mode="lines+markers",
            marker=dict(size=11, color="#f8fafc", symbol=symbols, line=dict(width=2, color="#0f172a")),
            line=dict(color="#f8fafc", width=3),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            connectgaps=False,
        )
    )
    fig.add_vrect(
        x0=2.5,
        x1=5.5,
        fillcolor="rgba(251, 113, 133, 0.28)",
        line=dict(color="rgba(251, 113, 133, 0.85)", width=1),
    )
    style(fig, height=460)
    fig.update_layout(
        title=f"{row.player} · {row.injury}",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        yaxis=dict(title="Match rating", range=[3.5, 10]),
        yaxis2=dict(
            title="Goal difference",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1,
        ),
        bargap=0.35,
    )
    st.plotly_chart(fig, theme=None)
    days = "ongoing" if row.ongoing else (f"{int(row.days_out)} days out" if pd.notna(row.days_out) else "duration unknown")
    drop = signed(row.gd_drop) if pd.notna(row.gd_drop) else "not available"
    change = signed(row.rating_change) if pd.notna(row.rating_change) else "not available"
    st.caption(f"{row.club} · {row.position} · {days}. Goal-difference drop {drop}. Rating change {change}.")


def leaderboard(frame: pd.DataFrame, minimum_episodes: int, show_gains: bool) -> None:
    title = "Comeback leaderboard" if show_gains else "Sharpest rating drops on return"
    st.subheader(title)
    st.caption(
        "A player needs a rating before the injury and a rating after return. "
        "The change is the after average minus the before average, then averaged across that player’s episodes."
    )
    rated = frame.dropna(subset=["rating_before", "rating_after"])
    if rated.empty:
        st.info("No filtered episodes have a before rating and an after rating.")
        return
    board = (
        rated.groupby("player", as_index=False)
        .agg(
            club=("club", most_common),
            position=("position", most_common),
            episodes=("player", "size"),
            rating_before=("rating_before", "mean"),
            rating_after=("rating_after", "mean"),
            rating_change=("rating_change", "mean"),
            gd_drop=("gd_drop", "mean"),
            median_days=("days_out", "median"),
            fifa_rating=("fifa_rating", "max"),
        )
    )
    board = board[board["episodes"] >= minimum_episodes]
    if board.empty:
        st.info(f"No player has at least {minimum_episodes} comparable episodes in this filter.")
        return
    board = board.sort_values("rating_change", ascending=not show_gains)
    view = board.head(15).copy()
    for column in ("rating_before", "rating_after", "rating_change", "gd_drop"):
        view[column] = view[column].round(2)
    view["median_days"] = view["median_days"].round(0)
    view["fifa_rating"] = view["fifa_rating"].round(0)
    view.columns = [
        "Player",
        "Club",
        "Position",
        "Episodes",
        "Rating before",
        "Rating after",
        "Rating change",
        "Mean GD drop",
        "Median days out",
        "FIFA rating",
    ]
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "Rating before": st.column_config.NumberColumn(format="%.2f"),
            "Rating after": st.column_config.NumberColumn(format="%.2f"),
            "Rating change": st.column_config.NumberColumn(format="%+.2f"),
            "Mean GD drop": st.column_config.NumberColumn(format="%+.2f"),
            "Median days out": st.column_config.NumberColumn(format="%.0f"),
            "FIFA rating": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    top = view.iloc[0]
    direction = "higher" if show_gains else "lower"
    st.caption(
        f"{top['Player']} rates {signed(top['Rating change'])} {direction} after return "
        f"across {int(top['Episodes'])} episodes."
    )


def repeat_chart(frame: pd.DataFrame) -> None:
    st.subheader("First absence and the next one")
    st.caption("Muscle and joint families only. Later absence means this player already had an earlier dated episode in the file.")
    muscle = frame[frame["injury_family"].isin(MUSCLE_FAMILIES)].dropna(subset=["rating_change"]).copy()
    if muscle.empty:
        st.info("No muscle or joint episodes in this filter have a rating change.")
        return
    muscle["spell"] = np.where(muscle["is_repeat"], "Later absence", "First recorded absence")
    summary = (
        muscle.groupby("spell", as_index=False)
        .agg(rating_change=("rating_change", "mean"), days=("days_out", "median"), episodes=("spell", "size"))
    )
    order = ["First recorded absence", "Later absence"]
    summary["spell"] = pd.Categorical(summary["spell"], order, ordered=True)
    summary = summary.sort_values("spell")
    fig = go.Figure(
        go.Bar(
            x=summary["spell"],
            y=summary["rating_change"],
            marker_color=[GAIN_COLOR, DROP_COLOR],
            customdata=np.column_stack([summary["episodes"], summary["days"]]),
            hovertemplate=(
                "%{x}<br>Mean rating change %{y:.2f}<br>%{customdata[0]} episodes"
                "<br>Median days %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_color="#cbd5e1", line_width=1)
    style(fig, height=380)
    fig.update_layout(title="Rating change, first absence against a later one", yaxis_title="Mean rating change")
    st.plotly_chart(fig, theme=None)


def _days_answer(frame: pd.DataFrame, include_subs: bool, minimum_type: int) -> str:
    usable = frame.dropna(subset=["days_out"])
    if usable.empty:
        return "None of the selected episodes have a usable length, so there is no days-lost ranking to give."
    totals = (
        usable.groupby("injury_family", as_index=False)["days_out"]
        .agg(days="sum", episodes="size", median_days="median")
        .sort_values("days", ascending=False)
    )
    counts = frame.groupby("injury_family").size().sort_values(ascending=False)
    heaviest = totals.iloc[0]
    frequent = counts.index[0]
    text = (
        f"**{heaviest['injury_family']}** removes the most days: "
        f"**{heaviest['days']:,.0f}** across {int(heaviest['episodes'])} episodes "
        f"with a usable length (median {heaviest['median_days']:.0f} days)."
    )
    text += (
        " Days lost is the burden of the absence, the time the player is actually missing. "
        "A family can be common and still take fewer days if each spell is short."
    )
    mentioned = {heaviest["injury_family"]}
    if frequent != heaviest["injury_family"]:
        frequent_days = totals.loc[totals["injury_family"] == frequent]
        frequent_total = float(frequent_days["days"].iloc[0]) if not frequent_days.empty else 0
        text += (
            f" **{frequent}** is the most common family here ({int(counts.iloc[0])} episodes) "
            f"and accounts for {frequent_total:,.0f} days."
        )
        mentioned.add(frequent)
    runner = totals.iloc[1] if len(totals) > 1 else None
    if runner is not None and runner["injury_family"] not in mentioned:
        text += (
            f" Next is **{runner['injury_family']}**, at {runner['days']:,.0f} days "
            f"across {int(runner['episodes'])} episodes."
        )
    return text


def _drop_answer(frame: pd.DataFrame, include_subs: bool, minimum_type: int) -> str:
    grouped = (
        frame.dropna(subset=["gd_drop"])
        .groupby("injury", as_index=False)
        .agg(drop=("gd_drop", "mean"), episodes=("gd_drop", "size"))
    )
    qualified = grouped[grouped["episodes"] >= minimum_type].sort_values("drop", ascending=False)
    if qualified.empty:
        return (
            f"No injury type in this selection has at least {minimum_type} episodes "
            "with a goal difference both before the injury and in the matches missed."
        )
    leader = qualified.iloc[0]
    text = (
        f"**{leader['injury']}** lines up with the largest move: a mean drop of **{signed(leader['drop'])}** "
        f"across {int(leader['episodes'])} episodes. "
        "The drop is the average goal difference before the injury, minus the average in the matches missed. "
        "A positive number means the team’s goal difference was worse while the player was out."
    )
    if leader["drop"] <= 0:
        text += " In this selection, even the top of the list is not a worse goal difference during the absence."
    if len(qualified) > 1:
        second = qualified.iloc[1]
        text += f" Next is **{second['injury']}** at {signed(second['drop'])} ({int(second['episodes'])} episodes)."
    text += (
        " This ranks injury types that occur often enough to compare. "
        "The opposition in those matches still sits inside the goal difference, so the number describes the team’s results during the absence, not a medical cause."
    )
    return text


def _age_fifa_answer(frame: pd.DataFrame, include_subs: bool, minimum_type: int) -> str:
    usable = frame.dropna(subset=["gd_drop", "age"])
    if len(usable) < 3:
        return "There are not enough episodes with both an age and a goal-difference drop to judge the relationship."
    age_corr = float(usable["age"].astype(float).corr(usable["gd_drop"]))
    rated = usable.dropna(subset=["fifa_rating"])
    fifa_corr = float(rated["fifa_rating"].corr(rated["gd_drop"])) if len(rated) >= 3 else np.nan
    text = (
        f"Age and the goal-difference drop move together only weakly. "
        f"Their correlation in this selection is **{age_corr:+.2f}** across {len(usable)} episodes. "
        "A value near zero means older players are not systematically the ones whose absence hurts the goal difference more."
    )
    if pd.notna(fifa_corr):
        high = rated[rated["fifa_rating"] >= 80]
        low = rated[rated["fifa_rating"] < 80]
        text += (
            f" The FIFA rating, which is the player’s season ability rather than a match rating, "
            f"has a correlation of **{fifa_corr:+.2f}** with the same drop."
        )
        if len(high) >= 5 and len(low) >= 5:
            text += (
                f" Players rated 80 or higher average a drop of {signed(high['gd_drop'].mean())} "
                f"({len(high)} episodes). Players rated below 80 average {signed(low['gd_drop'].mean())} "
                f"({len(low)} episodes)."
            )
        text += " The link is too slight to treat the FIFA rating as a way of sorting costly absences."
    return text


def _return_answer(frame: pd.DataFrame, include_subs: bool, minimum_type: int) -> str:
    rated = frame.dropna(subset=["rating_change"])
    if rated.empty:
        return "No selected episode has a match rating both before the injury and after return."
    improved = int((rated["rating_change"] > 0).sum())
    declined = int((rated["rating_change"] < 0).sum())
    flat = int((rated["rating_change"] == 0).sum())
    sub_note = (
        "Substitute ratings are included in these averages."
        if include_subs
        else "Substitute ratings, marked (S), are left out of these averages."
    )
    text = (
        f"Of **{len(rated)}** episodes with a rating on both sides of the absence, "
        f"**{improved}** come back higher, **{declined}** come back lower"
        + (f", and **{flat}** are unchanged" if flat else "")
        + f". The average change is {signed(rated['rating_change'].mean())}. "
        "Coming back into the side is a separate fact from rating higher than before the injury. "
        f"{sub_note}"
    )
    board = (
        rated.groupby("player", as_index=False)
        .agg(change=("rating_change", "mean"), episodes=("rating_change", "size"), club=("club", most_common))
    )
    comparable = board[board["episodes"] >= 2]
    pool = comparable if not comparable.empty else board
    best = pool.sort_values("change", ascending=False).iloc[0]
    worst = pool.sort_values("change", ascending=True).iloc[0]
    floor = "at least two comparable episodes" if not comparable.empty else "the episodes in this selection"
    if best["player"] != worst["player"]:
        text += (
            f" Among players with {floor}, **{best['player']}** ({best['club']}) has the largest gain, "
            f"{signed(best['change'])} across {int(best['episodes'])} episodes. "
            f"**{worst['player']}** ({worst['club']}) has the largest drop, "
            f"{signed(worst['change'])} across {int(worst['episodes'])} episodes."
        )
    return text


def _calendar_answer(frame: pd.DataFrame, include_subs: bool, minimum_type: int) -> str:
    usable = frame.dropna(subset=["injury_month"]).copy()
    if usable.empty:
        return "No selected episode has a usable injury date, so there is no calendar to read."
    usable["injury_month"] = usable["injury_month"].astype(int)
    by_month = usable.groupby("injury_month").size().sort_values(ascending=False)
    month_number = int(by_month.index[0])
    month_name = MONTHS[month_number - 1]
    counts = pd.crosstab(usable["club"], usable["injury_month"])
    stacked = counts.stack()
    club, cell_month = stacked.idxmax()
    text = (
        f"**{month_name}** is the busiest month in this selection, with **{int(by_month.iloc[0])}** episodes. "
        f"The single busiest club-month is **{club} in {MONTHS[int(cell_month) - 1]}**, "
        f"with {int(stacked.max())} episodes."
    )
    if len(by_month) > 1:
        second = int(by_month.index[1])
        text += f" {MONTHS[second - 1]} is next, with {int(by_month.iloc[1])} episodes."
    text += (
        " These are counts of recorded injury episodes, not a rate per match. "
        "Clubs in this file do not all cover the same seasons, so a high cell is a cluster in the records, not proof that one club is more injury-prone."
    )
    return text


def _repeat_answer(frame: pd.DataFrame, include_subs: bool, minimum_type: int) -> str:
    muscle = frame[frame["injury_family"].isin(MUSCLE_FAMILIES)].dropna(subset=["rating_change"]).copy()
    if muscle.empty:
        return "No muscle or joint episode in this selection has a rating both before the injury and after return."
    muscle["spell"] = np.where(muscle["is_repeat"], "Later absence", "First recorded absence")
    summary = muscle.groupby("spell", as_index=False).agg(
        change=("rating_change", "mean"),
        episodes=("rating_change", "size"),
    )
    days = (
        frame[frame["injury_family"].isin(MUSCLE_FAMILIES)]
        .dropna(subset=["days_out"])
        .assign(spell=lambda rows: np.where(rows["is_repeat"], "Later absence", "First recorded absence"))
        .groupby("spell")["days_out"]
        .median()
    )
    lookup = summary.set_index("spell")
    if "First recorded absence" not in lookup.index or "Later absence" not in lookup.index:
        only = summary.iloc[0]
        return (
            f"This selection only has **{only['spell'].lower()}** spells for muscle and joint injuries "
            f"({int(only['episodes'])} episodes, mean rating change {signed(only['change'])}). "
            "A later absence is one that follows an earlier dated episode for the same player."
        )
    first = lookup.loc["First recorded absence"]
    later = lookup.loc["Later absence"]
    first_days = days.get("First recorded absence", np.nan)
    later_days = days.get("Later absence", np.nan)
    text = (
        f"A later absence is a muscle or joint spell that follows an earlier dated episode for the same player. "
        f"The first recorded absence averages a rating change of **{signed(first['change'])}** "
        f"across {int(first['episodes'])} episodes. "
        f"A later absence averages **{signed(later['change'])}** across {int(later['episodes'])} episodes."
    )
    if pd.notna(first_days) and pd.notna(later_days):
        text += (
            f" Among every muscle or joint spell with a usable length, the median time out is "
            f"{first_days:.0f} days the first time and {later_days:.0f} days later."
        )
    gap = abs(float(later["change"]) - float(first["change"]))
    if gap < 0.1:
        text += (
            " The two groups sit close together. In this selection, a later muscle or joint absence "
            "does not look like a different kind of return from the first one on record."
        )
    elif float(later["change"]) > float(first["change"]):
        text += " Later absences come back a little higher, on average, than the first recorded one."
    else:
        text += " Later absences come back a little lower, on average, than the first recorded one."
    return text


QUESTION_ANSWERS = [
    ("Which injury families remove the most days?", _days_answer),
    ("Which injury types coincide with the biggest fall in team goal difference?", _drop_answer),
    ("Do age and FIFA rating line up with a larger drop?", _age_fifa_answer),
    ("Who returns above their pre-injury rating, and who does not?", _return_answer),
    ("Which club-months are the busiest?", _calendar_answer),
    ("Do later muscle and joint absences look different from a player’s first recorded one?", _repeat_answer),
]


def question_answers(frame: pd.DataFrame, include_subs: bool, minimum_type: int) -> None:
    st.subheader("Questions")
    st.caption("Choose a question. The answer is worked out from the episodes selected in the sidebar.")
    labels = [label for label, _answer in QUESTION_ANSWERS]
    choice = st.selectbox(
        "Choose a question",
        labels,
        index=None,
        placeholder="Choose a question",
    )
    if choice is None:
        return
    answer = dict(QUESTION_ANSWERS)[choice](frame, include_subs, minimum_type)
    st.markdown(answer)


def _phase_mean(frame: pd.DataFrame, phase: str, kind: str) -> pd.Series:
    parts = [frame[f"m{index}_{phase}_{kind}"] for index in (1, 2, 3)]
    return pd.concat(parts, axis=1).mean(axis=1)


def notebook_charts(frame: pd.DataFrame) -> None:
    """The seven charts from the analysis notebook, on the filtered episodes.

    Rating averages count the number written on a substitute appearance, which
    is how that notebook was calculated. Duration is return date minus injury
    date, so the two spells that run backwards stay in the age-group average.
    """
    st.header("Figures from the analysis notebook")
    st.caption(
        "Injury counts, rating change, the five largest team drops, recovery by age group, "
        "and goal difference before, during and after the absence. "
        "A substitute rating is counted as the number written beside (S). Sidebar filters apply."
    )
    if frame.empty:
        st.info("No episodes match the current filters.")
        return

    work = frame.copy()
    work["nb_rating_before"] = _phase_mean(work, "before", "rating")
    work["nb_rating_after"] = _phase_mean(work, "after", "rating")
    work["nb_days"] = (work["return_date"] - work["injury_date"]).dt.days

    players_col, clubs_col = st.columns(2)
    with players_col:
        _notebook_player_counts(work)
    with clubs_col:
        _notebook_club_counts(work)

    _notebook_rating_swing(work)

    impact_col, compare_col = st.columns(2)
    with impact_col:
        _notebook_top_impacts(work)
    with compare_col:
        _notebook_rating_comparison(work)

    age_col, phase_col = st.columns(2)
    with age_col:
        _notebook_recovery_by_age(work)
    with phase_col:
        _notebook_phase_goal_difference(work)


def _notebook_player_counts(frame: pd.DataFrame) -> None:
    st.subheader("Most frequently injured players")
    counts = (
        frame.groupby("player", as_index=False)
        .size()
        .rename(columns={"size": "injuries"})
        .sort_values(["injuries", "player"], ascending=[False, True])
        .head(10)
        .sort_values(["injuries", "player"], ascending=[True, False])
    )
    st.caption("Each bar is the number of injury episodes recorded for that player in this filter.")
    fig = go.Figure(
        go.Bar(
            x=counts["injuries"],
            y=counts["player"],
            orientation="h",
            marker_color="#38bdf8",
            hovertemplate="%{y}<br>%{x} episodes<extra></extra>",
        )
    )
    style(fig, height=480, margin_l=168)
    fig.update_layout(
        title="Top 10 most frequently injured players",
        xaxis_title="Number of injuries",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=counts["player"].tolist())
    st.plotly_chart(fig, theme=None)
    top_count = int(counts["injuries"].max())
    leaders = sorted(counts.loc[counts["injuries"] == top_count, "player"])
    names = " and ".join(leaders) if len(leaders) < 3 else ", ".join(leaders)
    st.caption(f"{names} {'have' if len(leaders) > 1 else 'has'} the most episodes in this filter ({top_count}).")


def _notebook_club_counts(frame: pd.DataFrame) -> None:
    st.subheader("Clubs with the most injury cases")
    counts = (
        frame.groupby("club", as_index=False)
        .size()
        .rename(columns={"size": "injuries"})
        .sort_values(["injuries", "club"], ascending=[True, False])
    )
    st.caption("Episode counts by club. Eight clubs are in the file, so this is the full list.")
    fig = go.Figure(
        go.Bar(
            x=counts["injuries"],
            y=counts["club"],
            orientation="h",
            marker_color="#38bdf8",
            hovertemplate="%{y}<br>%{x} episodes<extra></extra>",
        )
    )
    style(fig, height=480, margin_l=120)
    fig.update_layout(
        title="Clubs with the most injury cases",
        xaxis_title="Number of injuries",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=counts["club"].tolist())
    st.plotly_chart(fig, theme=None)
    leader = counts.iloc[-1]
    st.caption(f"{leader['club']} has the most episodes in this filter ({int(leader['injuries'])}).")


def _notebook_rating_swing(frame: pd.DataFrame) -> None:
    st.subheader("Highest rating improvement and decline")
    st.caption(
        "A player’s change is their average rating after return, minus their average rating before injury, "
        "across the episodes in this filter. The five largest gains and the five largest drops are shown."
    )
    summary = frame.groupby("player", as_index=False).agg(
        rating_before=("nb_rating_before", "mean"),
        rating_after=("nb_rating_after", "mean"),
    )
    summary["rating_change"] = summary["rating_after"] - summary["rating_before"]
    valid = summary.dropna(subset=["rating_before", "rating_after"])
    improved = valid[valid["rating_change"] > 0].sort_values("rating_change", ascending=False).head(5)
    declined = valid[valid["rating_change"] < 0].sort_values("rating_change", ascending=True).head(5)
    combined = pd.concat([improved, declined], ignore_index=True)
    if combined.empty:
        st.info("No player in this filter has a rating both before the injury and after return.")
        return
    # Largest gain at the top, largest drop at the bottom.
    combined = combined.sort_values("rating_change", ascending=True)
    colors = [GAIN_COLOR if value > 0 else DROP_COLOR for value in combined["rating_change"]]
    fig = go.Figure(
        go.Bar(
            x=combined["rating_change"],
            y=combined["player"],
            orientation="h",
            marker_color=colors,
            customdata=np.column_stack([combined["rating_before"], combined["rating_after"]]),
            hovertemplate=(
                "%{y}<br>Change %{x:.2f}"
                "<br>Before %{customdata[0]:.2f}<br>After %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="#cbd5e1", line_width=1)
    style(fig, height=480, margin_l=150)
    fig.update_layout(
        title="Players with the highest rating improvement and decline",
        xaxis_title="Change in average player rating",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=combined["player"].tolist())
    st.plotly_chart(fig, theme=None)
    best = combined.iloc[-1]
    worst = combined.iloc[0]
    sentences = []
    if best["rating_change"] > 0:
        sentences.append(f"{best['player']} has the largest gain ({signed(best['rating_change'])}).")
    if worst["rating_change"] < 0:
        sentences.append(f"{worst['player']} has the largest drop ({signed(worst['rating_change'])}).")
    st.caption(" ".join(sentences))


def _notebook_top_impacts(frame: pd.DataFrame) -> None:
    st.subheader("Largest team-performance drops")
    st.caption(
        "Each bar is one injury episode: mean goal difference before the injury, "
        "minus mean goal difference in the matches missed. The five largest values are shown."
    )
    top = (
        frame.dropna(subset=["gd_drop"])
        .sort_values(["gd_drop", "player"], ascending=[False, True])
        .head(5)
        .sort_values(["gd_drop", "player"], ascending=[True, False])
    )
    if top.empty:
        st.info("No filtered episodes have a goal difference before the injury and during the absence.")
        return
    labels = top["player"].tolist()
    if len(labels) != len(set(labels)):
        labels = [f"{player} · {injury}" for player, injury in zip(top["player"], top["injury"])]
    fig = go.Figure(
        go.Bar(
            x=top["gd_drop"],
            y=labels,
            orientation="h",
            marker_color=DROP_COLOR,
            customdata=np.column_stack([top["injury"], top["club"]]),
            hovertemplate="%{y}<br>%{customdata[0]} · %{customdata[1]}<br>Drop %{x:.2f}<extra></extra>",
        )
    )
    style(fig, height=420, margin_l=160)
    fig.update_layout(
        title="Top 5 injuries with the highest team-performance impact",
        xaxis_title="Team performance drop index",
        yaxis_title="",
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=labels)
    st.plotly_chart(fig, theme=None)
    leader = top.iloc[-1]
    st.caption(
        f"{leader['player']} · {leader['injury']} ({leader['club']}) "
        f"has the largest drop in this filter ({signed(leader['gd_drop'])})."
    )


def _notebook_rating_comparison(frame: pd.DataFrame) -> None:
    st.subheader("Rating before and after injury")
    st.caption("Mean match rating across every filtered episode, before the injury and after return.")
    before = frame["nb_rating_before"].mean()
    after = frame["nb_rating_after"].mean()
    if pd.isna(before) and pd.isna(after):
        st.info("No filtered episodes have a match rating before or after the injury.")
        return
    fig = go.Figure(
        go.Bar(
            x=["Before injury", "After injury"],
            y=[before, after],
            marker_color=["#94a3b8", "#38bdf8"],
            hovertemplate="%{x}<br>Average rating %{y:.2f}<extra></extra>",
        )
    )
    style(fig, height=420, margin_l=64)
    fig.update_layout(
        title="Overall average player rating before vs after injury",
        xaxis_title="",
        yaxis_title="Average player rating",
        showlegend=False,
    )
    st.plotly_chart(fig, theme=None)
    if pd.notna(before) and pd.notna(after):
        st.caption(
            f"Average rating moves from {before:.2f} before the injury to {after:.2f} after return "
            f"({signed(after - before)})."
        )


def _notebook_recovery_by_age(frame: pd.DataFrame) -> None:
    st.subheader("Recovery time by age group")
    st.caption("Mean days from injury date to return. Open absences and unusable dates are left out.")
    usable = frame.dropna(subset=["age", "nb_days"]).copy()
    if usable.empty:
        st.info("No filtered episodes have both an age and a usable duration.")
        return
    labels = ["Under 21", "21-25", "26-30", "31-40", "Above 40"]
    usable["age_group"] = pd.cut(
        usable["age"],
        bins=[0, 21, 25, 30, 40, 100],
        labels=labels,
    )
    trends = (
        usable.groupby("age_group", observed=True)["nb_days"]
        .agg(days="mean", episodes="size")
        .reindex(labels)
        .dropna(subset=["days"])
        .reset_index()
    )
    fig = go.Figure(
        go.Bar(
            x=trends["age_group"].astype(str),
            y=trends["days"],
            marker_color="#38bdf8",
            customdata=trends["episodes"],
            hovertemplate="%{x}<br>%{y:.1f} days<br>%{customdata:.0f} episodes<extra></extra>",
        )
    )
    style(fig, height=420, margin_l=96)
    fig.update_layout(
        title="Average injury recovery duration by age group",
        xaxis_title="Age group",
        yaxis_title="Average recovery duration (days)",
        showlegend=False,
    )
    st.plotly_chart(fig, theme=None)
    longest = trends.loc[trends["days"].idxmax()]
    st.caption(
        f"{longest['age_group']} has the longest average absence in this filter "
        f"({longest['days']:.0f} days across {int(longest['episodes'])} episodes)."
    )


def _notebook_phase_goal_difference(frame: pd.DataFrame) -> None:
    st.subheader("Goal difference across the absence")
    st.caption("Mean team goal difference in the matches before the injury, the matches missed, and the matches after return.")
    values = [
        frame["gd_before"].mean(),
        frame["gd_missed"].mean(),
        frame["gd_after"].mean(),
    ]
    labels = ["Before injury", "During injury", "After injury"]
    if all(pd.isna(value) for value in values):
        st.info("No filtered episodes have a goal difference in any of the three windows.")
        return
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#94a3b8", DROP_COLOR, GAIN_COLOR],
            hovertemplate="%{x}<br>Average goal difference %{y:.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#cbd5e1", line_width=1)
    style(fig, height=420, margin_l=88)
    fig.update_layout(
        title="Team performance before, during and after injury",
        xaxis_title="Injury phase",
        yaxis_title="Average goal difference",
        showlegend=False,
    )
    st.plotly_chart(fig, theme=None)
    if all(pd.notna(value) for value in values):
        st.caption(
            f"Average goal difference is {signed(values[0])} before, {signed(values[1])} during the absence, "
            f"and {signed(values[2])} after return."
        )


def brand_header() -> None:
    encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode()
    st.markdown(
        f"""
        <div class="brand">
            <img src="data:image/png;base64,{encoded}" alt="Premier League ball">
            <h1>FootLens Analytics</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    base, _stats = get_data()
    brand_header()
    st.caption(
        "Use this to see which absences move the goal difference, "
        "who comes back above their pre-injury rating, and where the injury calendar bunches up."
    )

    filtered, include_subs, clubs = filter_episodes(base)
    club_wallpaper(clubs)

    comparable_ratings = int(filtered[["rating_before", "rating_after"]].notna().all(axis=1).sum())
    median_days = filtered["days_out"].median()
    mean_drop = filtered["gd_drop"].mean()
    mean_change = filtered["rating_change"].mean()

    left, mid_left, middle, mid_right, right = st.columns(5)
    left.metric("Episodes", f"{len(filtered)}", help="Injury spells that pass the sidebar filters.")
    mid_left.metric("Players", f"{filtered['player'].nunique()}", help="Distinct players inside the filtered episodes.")
    middle.metric(
        "Median days out",
        "—" if pd.isna(median_days) else f"{median_days:.0f}",
        help="Median days from injury date to return, ignoring open and unusable dates.",
    )
    mid_right.metric(
        "Mean GD drop",
        "—" if pd.isna(mean_drop) else signed(mean_drop),
        help="Average of (goal difference before injury minus goal difference while the player was out). Positive means a worse goal difference during the absence.",
    )
    right.metric(
        "Mean rating change",
        "—" if pd.isna(mean_change) else signed(mean_change),
        help=f"Average of (rating after return minus rating before injury), across {comparable_ratings} episodes with both.",
    )

    minimum_type = st.slider(
        "Minimum episodes for an injury type to enter the ranking",
        min_value=3,
        max_value=15,
        value=5,
        help="Stops a single unusual spell from topping the injury chart.",
    )
    drop_col, burden_col = st.columns(2)
    with drop_col:
        injury_drop_chart(filtered, minimum_type)
    with burden_col:
        burden_chart(filtered)

    age_scatter(filtered)

    heat_col, club_col = st.columns(2)
    with heat_col:
        heatmap(filtered)
    with club_col:
        club_drop_chart(filtered)

    timeline(filtered, include_subs)

    board_left, board_right = st.columns([1.3, 1])
    with board_left:
        show_gains = st.radio(
            "Rank players by",
            ["Rating gain on return", "Rating drop on return"],
            horizontal=True,
        )
        minimum_player = st.slider("Minimum comparable episodes per player", 1, 5, 2)
        leaderboard(filtered, minimum_player, show_gains.startswith("Rating gain"))
    with board_right:
        repeat_chart(filtered)

    notebook_charts(filtered)

    st.download_button(
        "Download the filtered episodes",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_injury_episodes.csv",
        mime="text/csv",
    )
    question_answers(filtered, include_subs, minimum_type)


if __name__ == "__main__":
    main()
