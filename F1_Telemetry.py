
import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROMANIA_TZ = ZoneInfo("Europe/Bucharest")  # EET/EEST - zoneinfo resolves the DST
# switch automatically, so a fixed UTC+2/+3 offset is never hard-coded anywhere below.

st.set_page_config(page_title="F1 Telemetry", layout="wide")
st.title("F1 Driver Racing Lines")

# --- 1. Initialize FastF1 Cache & Plotting Styles ---
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')
fastf1.plotting.setup_mpl(mpl_timedelta_support=False, color_scheme='fastf1')

BG_COLOR = '#0E0E14'
TRACK_COLOR = '#2D2D38'
CORNER_MARKER_COLOR = '#2A2A35'
CORRIDOR_COLOR = '#3A3A55'
VIOLATION_COLOR = '#FF3B30'
SECTOR_COLORS = {
    'Start/Finish': '#FF1801',
    'Sector 1 Split': '#00E5FF',
    'Sector 2 Split': '#FFD600',
}
TRACK_LIMIT_BIN_SIZE_M = 5.0
DASH_OPTIONS = ['solid', 'dash', 'dot', 'dashdot']  # teammates share a team color, so a repeat
# color within the selected drivers gets a different dash pattern instead, to stay distinguishable.


MAX_PLAUSIBLE_SPEED_MPS = 130.0  # ~468 km/h, safely above any real F1 speed
XY_UNITS_PER_METRE = 10.0  # FastF1 X/Y/Z are in decimetres; Distance/Time are not - confirmed by
# comparing summed raw XY step length over a whole lap to that lap's Distance channel: ratio is
# exactly 10.00 (checked on 2026 Austria). Every meter-denominated threshold below converts through
# this before comparing against raw X/Y deltas.


def _position_glitch_mask(tel, min_xy_step=1.5, min_distance_step=3.0):
    """Return a boolean mask (len == len(tel)) flagging rows corrupted by a GPS
    position-feed resync glitch, seen on every lap checked (2024 and 2026 data
    alike, worst in 2026 Hungary): the raw position channel holds a stale
    coordinate for several samples while Distance/Speed keep advancing normally,
    then snaps toward the true position and makes a few decaying over-fast
    corrective jumps before settling. Both phases inject a physically-impossible
    step into any racing-line plot or track-limit corridor. Two independent,
    purely LOCAL checks (each row compared only to its immediate predecessor,
    never a persisted "last good" anchor) - so once the signal settles back to a
    plausible step, the rest of the lap is kept; a genuinely slow corner has small
    steps in every channel together and trips neither check:
    - stale/frozen: X,Y barely moves while Distance keeps climbing.
    - resync tail: implied speed from the previous raw sample is impossible.
    """
    if len(tel) < 3:
        return np.zeros(len(tel), dtype=bool)
    x, y, dist = tel['X'].to_numpy(), tel['Y'].to_numpy(), tel['Distance'].to_numpy()
    time_col = 'Time' if 'Time' in tel.columns else 'SessionTime'
    t_sec = tel[time_col].to_numpy() / np.timedelta64(1, 's')

    xy_step_m = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) / XY_UNITS_PER_METRE
    dist_step = np.diff(dist)
    dt = np.diff(t_sec)
    with np.errstate(divide='ignore', invalid='ignore'):
        implied_speed = np.where(dt > 0, xy_step_m / dt, 0.0)

    stuck = (xy_step_m < min_xy_step) & (dist_step > min_distance_step)
    resync_tail = implied_speed > MAX_PLAUSIBLE_SPEED_MPS
    bad_step = stuck | resync_tail

    mask = np.zeros(len(tel), dtype=bool)
    mask[1:] = bad_step

    # Row 0 has no earlier row within this lap to check it against, so a glitch
    # already in progress at the very first sample was never flagged - the bad
    # step (0->1) always blamed row 1 instead. Use row 2 as a tie-breaker: if
    # skipping row 1 entirely (0->2) is STILL an impossible jump, row 1 was
    # innocent and row 0 itself is the corrupted one - flag it instead.
    if len(tel) >= 3 and bad_step[0]:
        step02_m = np.hypot(x[2] - x[0], y[2] - y[0]) / XY_UNITS_PER_METRE
        dt02 = t_sec[2] - t_sec[0]
        if dt02 > 0 and step02_m / dt02 > MAX_PLAUSIBLE_SPEED_MPS:
            mask[0] = True
            mask[1] = False
    return mask


def _drop_stuck_position_samples(tel):
    mask = _position_glitch_mask(tel)
    if not mask.any():
        return tel
    return tel[~mask].reset_index(drop=True)

# --- 2. Dynamic Year, Race, and Session Selectors ---
col1, col2, col3 = st.columns(3)
with col1:
    current_year = datetime.now().year
    available_years = list(range(current_year, 2022, -1))
    selected_year = st.selectbox("Season", available_years)

with col2:
    @st.cache_data(ttl=86400, show_spinner=False)
    def get_event_schedule(year):
        schedule = fastf1.get_event_schedule(year)
        return schedule[schedule['EventFormat'] != 'testing']['EventName'].tolist()

    races = get_event_schedule(selected_year)
    selected_race = st.selectbox("Grand Prix", races)

with col3:
    session_dict = {"Race": "R", "Sprint": "S", "Qualifying": "Q"}
    selected_session_type = st.selectbox("Session", list(session_dict.keys()), index=0)


@st.cache_data(ttl=86400, show_spinner=False)
def get_full_event_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    return schedule[schedule['EventFormat'] != 'testing']


def build_calendar_df(schedule, include_practice):
    """Long-format calendar: one row per session, with the UTC schedule (fastf1's
    Session{n}DateUtc) converted to Romania local time. Session slot numbering/names
    differ between conventional and sprint weekends (e.g. Session2 is Practice 2 on a
    conventional weekend but Sprint Qualifying on a sprint one) - Session{n} is read
    generically per-row rather than assumed to always be a fixed session type.
    """
    now_utc = pd.Timestamp.now(tz='UTC')
    rows = []
    for _, event in schedule.iterrows():
        for i in range(1, 6):
            name = event.get(f'Session{i}')
            date_utc = event.get(f'Session{i}DateUtc')
            if pd.isna(name) or pd.isna(date_utc):
                continue
            if not include_practice and str(name).startswith('Practice'):
                continue
            dt_utc = pd.Timestamp(date_utc, tz='UTC')
            dt_ro = dt_utc.tz_convert(ROMANIA_TZ)
            rows.append({
                'Round': int(event['RoundNumber']),
                'Grand Prix': event['EventName'],
                'Session': name,
                'Date (Romania)': dt_ro.strftime('%a %d %b %Y'),
                'Time (Romania)': dt_ro.strftime('%H:%M'),
                'Status': 'Completed' if dt_utc < now_utc else '',
                '_sort': dt_utc,
            })
    df = pd.DataFrame(rows).sort_values('_sort').reset_index(drop=True)
    if not df.empty:
        next_idx = df.index[df['_sort'] >= now_utc]
        if len(next_idx):
            df.loc[next_idx[0], 'Status'] = 'Next up'
    return df.drop(columns='_sort')


def render_race_calendar(year):
    schedule = get_full_event_schedule(year)
    if schedule.empty:
        st.caption(f"No race calendar data available for {year}.")
        return
    include_practice = st.checkbox("Show practice sessions", value=False, key=f"cal_practice_{year}")
    calendar_df = build_calendar_df(schedule, include_practice)
    if calendar_df.empty:
        st.caption("No sessions to show.")
        return

    next_up = calendar_df[calendar_df['Status'] == 'Next up']
    if not next_up.empty:
        row = next_up.iloc[0]
        st.info(f"Next up: **{row['Grand Prix']} - {row['Session']}** on "
                f"{row['Date (Romania)']} at {row['Time (Romania)']} (Romania time)")

    st.dataframe(
        calendar_df.style.apply(
            lambda r: ['background-color: #2A4A2A' if r['Status'] == 'Next up' else '' for _ in r],
            axis=1,
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("All times shown are local to Romania (Europe/Bucharest) and account for daylight saving.")


with st.expander("📅 Race Calendar (Romania Time)", expanded=False):
    render_race_calendar(selected_year)


# --- 3. Cached Data Access Layer ---
# Session objects are large and mutable, so they're cached as a resource (kept live in
# memory by reference) rather than as data (which would pickle/copy them on every access).
@st.cache_resource(show_spinner=False)
def get_session_data(year, race, session_type):
    session = fastf1.get_session(year, race, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    # FastF1's load() can return normally even when one data type (laps, here) failed
    # to populate for this specific session - accessing it later then throws
    # DataNotLoadedError deep inside the dashboard instead of here. Checking now, and
    # raising if it's missing, matters for more than a cleaner error: since this
    # function is st.cache_resource-cached, a successful return caches the session
    # object as "good" even with laps missing, so every future call for this same
    # (year, race, session_type) would hand back the same broken object and fail the
    # same way forever, until the server process restarts. Raising here instead means
    # cache_resource does NOT cache it, so the next attempt gets a fresh retry.
    try:
        _ = session.laps
    except fastf1.exceptions.DataNotLoadedError as e:
        raise RuntimeError(
            f"Lap data isn't available for {race} {year} ({session_type}) - it may not be "
            f"published yet, or the data source had trouble loading it. Try again, or pick "
            f"a different session."
        ) from e
    return session


@st.cache_data(show_spinner=False, persist="disk")
def get_reference_lap_telemetry(year, race, session_type):
    session = get_session_data(year, race, session_type)
    ref_lap = session.laps.pick_fastest()
    raw_tel = ref_lap.get_telemetry().reset_index(drop=True)[['X', 'Y', 'Speed', 'Distance', 'SessionTime']]
    glitch_mask = _position_glitch_mask(raw_tel)
    glitch_fraction = float(glitch_mask.mean()) if len(raw_tel) else 0.0
    ref_tel = raw_tel[~glitch_mask].reset_index(drop=True)
    sf_time = ref_lap['LapStartTime'] if pd.notna(ref_lap['LapStartTime']) else ref_tel['SessionTime'].iloc[0]
    return ref_tel, sf_time, ref_lap['Sector1SessionTime'], ref_lap['Sector2SessionTime'], glitch_fraction


@st.cache_data(show_spinner=False, persist="disk")
def get_driver_color_cached(year, race, session_type, driver):
    session = get_session_data(year, race, session_type)
    try:
        return fastf1.plotting.get_driver_color(driver, session=session)
    except Exception:
        return '#FFFFFF'


@st.cache_data(show_spinner=False, persist="disk")
def get_compound_color_cached(year, race, session_type, compound):
    session = get_session_data(year, race, session_type)
    try:
        return fastf1.plotting.get_compound_color(compound, session=session)
    except Exception:
        return '#FFFFFF'


@st.cache_data(show_spinner=False, persist="disk")
def get_circuit_corners(year, race, session_type):
    session = get_session_data(year, race, session_type)
    circuit_info = session.get_circuit_info()
    return circuit_info.corners if circuit_info else None


@st.cache_data(show_spinner=False, persist="disk")
def get_single_lap_telemetry(year, race, session_type, driver, lap_number):
    """Full telemetry for exactly one quick lap of one driver - X/Y/Speed/Distance/
    Throttle/Brake plus that lap's TyreLife - cached individually per lap rather than
    per requested-laps TUPLE like the old get_driver_lap_telemetry was. This is the
    shared fetch primitive behind get_driver_lap_telemetry, get_stint_telemetry,
    get_average_lap_trend, and the track-limit corridor build, so:
    - overlapping lap selections (dragging a range, or switching between Single Lap /
      Average Trend / a stint that shares laps already viewed) reuse whatever was
      already fetched instead of re-fetching the whole batch from scratch.
    - the same lap viewed from two different charts shares one fetch instead of each
      chart doing its own redundant lap.get_telemetry() call.
    """
    session = get_session_data(year, race, session_type)
    driver_laps = session.laps.pick_driver(driver).pick_quicklaps()
    matching = driver_laps[driver_laps['LapNumber'] == lap_number]
    for _, lap in matching.iterlaps():
        tel = _drop_stuck_position_samples(lap.get_telemetry())[
            ['X', 'Y', 'Speed', 'Distance', 'Throttle', 'Brake']].copy()
        tel['LapNumber'] = lap['LapNumber']
        tel['TyreLife'] = lap['TyreLife']
        return tel
    return pd.DataFrame(columns=['X', 'Y', 'Speed', 'Distance', 'Throttle', 'Brake', 'LapNumber', 'TyreLife'])


def _fetch_laps(fetch_args):
    """fetch_args: list of (year, race, session_type, driver, lap_number) tuples.
    Sequential, deliberately - a thread pool was tried here and measured 27% SLOWER
    than sequential fetching (11.66s vs 14.84s for 15 laps, checked against real
    cached 2026 Austria data), not faster: FastF1's get_telemetry() apparently
    doesn't release the GIL enough during its merge/interpolate work for threads to
    add real concurrency, so they only add context-switch overhead. The actual win
    here is that each lap is independently st.cache_data-cached via
    get_single_lap_telemetry, so overlapping lap selections across calls (or across
    different charts viewing the same lap) reuse what's already fetched.
    """
    return [get_single_lap_telemetry(*args) for args in fetch_args]


def get_driver_lap_telemetry(year, race, session_type, driver, lap_numbers):
    fetch_args = [(year, race, session_type, driver, n) for n in lap_numbers]
    frames = [f for f in _fetch_laps(fetch_args) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=['X', 'Y', 'Speed', 'Distance', 'Throttle', 'Brake', 'LapNumber'])
    return pd.concat(frames, ignore_index=True)[['X', 'Y', 'Speed', 'Distance', 'Throttle', 'Brake', 'LapNumber']]


AVERAGE_TREND_BIN_SIZE_M = 10.0


@st.cache_data(show_spinner="Averaging laps...", persist="disk")
def get_average_lap_trend(year, race, session_type, driver):
    """Mean X/Y/Speed at each distance-along-lap bin, across every one of the
    driver's quick laps - a single representative "trend" line instead of an
    overlay of every lap. Reuses the same distance-binning idea as the track-limit
    corridor (get_track_limit_envelope), just averaged instead of min/max'd, and
    per-driver rather than field-wide.

    Bins outside a given lap's own recorded distance range are left NaN (not
    clipped to that lap's boundary value, unlike the corridor) and excluded via
    nanmean - clipping would drag a bin's average toward a lap's edge value it
    never actually reached, which is fine for widening/narrowing a corridor but
    would corrupt an averaged position/speed.
    """
    session = get_session_data(year, race, session_type)
    driver_laps = session.laps.pick_driver(driver).pick_quicklaps()
    if driver_laps.empty:
        return pd.DataFrame(columns=['X', 'Y', 'Speed', 'Distance'])

    fetch_args = [(year, race, session_type, driver, n)
                  for n in driver_laps['LapNumber'].astype(int).tolist()]
    lap_tels, max_dist = [], 0.0
    for tel in _fetch_laps(fetch_args):
        if tel.empty:
            continue
        lap_tels.append(tel[['X', 'Y', 'Speed', 'Distance']])
        max_dist = max(max_dist, tel['Distance'].max())

    if not lap_tels:
        return pd.DataFrame(columns=['X', 'Y', 'Speed', 'Distance'])

    bins = np.arange(0, max_dist, AVERAGE_TREND_BIN_SIZE_M)
    x_stack, y_stack, speed_stack = [], [], []
    for tel in lap_tels:
        d = tel['Distance'].to_numpy()
        order = np.argsort(d)
        d = d[order]
        x_stack.append(np.interp(bins, d, tel['X'].to_numpy()[order], left=np.nan, right=np.nan))
        y_stack.append(np.interp(bins, d, tel['Y'].to_numpy()[order], left=np.nan, right=np.nan))
        speed_stack.append(np.interp(bins, d, tel['Speed'].to_numpy()[order], left=np.nan, right=np.nan))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        mean_x = np.nanmean(np.vstack(x_stack), axis=0)
        mean_y = np.nanmean(np.vstack(y_stack), axis=0)
        mean_speed = np.nanmean(np.vstack(speed_stack), axis=0)

    return pd.DataFrame(
        {'Distance': bins, 'X': mean_x, 'Y': mean_y, 'Speed': mean_speed}
    ).dropna().reset_index(drop=True)


@st.cache_data(show_spinner=False, persist="disk")
def get_driver_stints(year, race, session_type, driver):
    session = get_session_data(year, race, session_type)
    driver_laps = session.laps.pick_driver(driver).pick_quicklaps()
    if driver_laps.empty:
        return pd.DataFrame(columns=['Stint', 'Compound', 'LapStart', 'LapEnd'])
    stints = (driver_laps.groupby('Stint')
              .agg(Compound=('Compound', 'first'),
                   LapStart=('LapNumber', 'min'),
                   LapEnd=('LapNumber', 'max'))
              .reset_index())
    return stints


@st.cache_data(show_spinner=False, persist="disk")
def get_stint_telemetry(year, race, session_type, driver, stint):
    """Throttle/Brake telemetry for every lap in one stint, tagged with that lap's
    TyreLife, so brake/throttle points can be compared across the life of a tyre.
    Built from the same per-lap cache as get_driver_lap_telemetry/
    get_average_lap_trend, so a lap already viewed on the track map or in Average
    Trend mode doesn't get re-fetched here.
    """
    session = get_session_data(year, race, session_type)
    driver_laps = session.laps.pick_driver(driver).pick_quicklaps()
    stint_laps = driver_laps[driver_laps['Stint'] == stint]
    fetch_args = [(year, race, session_type, driver, n)
                  for n in stint_laps['LapNumber'].astype(int).tolist()]
    frames = [f for f in _fetch_laps(fetch_args) if not f.empty]
    if not frames:
        return pd.DataFrame(columns=['Distance', 'Throttle', 'Brake', 'LapNumber', 'TyreLife'])
    return pd.concat(frames, ignore_index=True)[['Distance', 'Throttle', 'Brake', 'LapNumber', 'TyreLife']]


FINISHED_STATUSES = {'Finished', 'Lapped'}  # everything else in session.results['Status']
# (Retired, Accident, Disqualified, Did not start, ...) is a retirement for the position chart.


@st.cache_data(show_spinner=False, persist="disk")
def get_position_progression(year, race, session_type):
    """Per-lap running position for every driver, plus each driver's actual starting
    GRID position (from session.results - reflects any penalties/pit-lane starts,
    not just qualifying classification) and finishing status. Only meaningful for
    Race/Sprint sessions - Qualifying doesn't have a persistent running position.
    """
    session = get_session_data(year, race, session_type)
    laps = session.laps[['Driver', 'LapNumber', 'Position']].dropna(subset=['Position']).copy()
    laps['LapNumber'] = laps['LapNumber'].astype(int)
    laps['Position'] = laps['Position'].astype(int)

    results = session.results[['Abbreviation', 'GridPosition', 'Status']].copy()
    results = results.rename(columns={'Abbreviation': 'Driver'})
    results['GridPosition'] = pd.to_numeric(results['GridPosition'], errors='coerce')

    return laps, results


def _format_laptime(td):
    if pd.isna(td):
        return '-'
    total_seconds = td.total_seconds()
    minutes, seconds = divmod(total_seconds, 60)
    return f"{int(minutes)}:{seconds:06.3f}"


def _format_gap(td):
    if pd.isna(td):
        return '-'
    if td.total_seconds() < 0.0005:
        return 'Leader'
    return f"+{td.total_seconds():.3f}s"


@st.cache_data(show_spinner=False)
def get_lap_status_table(year, race, session_type, lap_number):
    """Per-driver race status at one lap, in running order: tyre compound, gap to
    leader/car-in-front (both derived from session.laps' cumulative 'Time' column,
    compared at the same LapNumber across drivers - the same simplification
    get_position_progression uses, so it's inconsistent-with-lapped-cars in exactly
    the same way, not a new source of error), that lap's time, and each driver's own
    fastest lap so far. Returns (display_df, fastest_mask) - fastest_mask flags which
    row(s) hold the outright fastest lap of the race up to this point, for highlighting.
    """
    session = get_session_data(year, race, session_type)
    laps = session.laps

    lap_rows = laps[laps['LapNumber'] == lap_number].dropna(subset=['Time']).copy()
    if lap_rows.empty:
        return pd.DataFrame(), []

    laps_upto = laps[laps['LapNumber'] <= lap_number]
    fastest_so_far = laps_upto.groupby('Driver')['LapTime'].min()
    overall_fastest = fastest_so_far.min() if not fastest_so_far.empty else pd.NaT

    lap_rows = lap_rows.sort_values('Position', na_position='last').reset_index(drop=True)
    ranked = lap_rows[lap_rows['Position'].notna()]
    leader_time = ranked['Time'].iloc[0] if not ranked.empty else pd.NaT

    records = []
    prev_time = None
    for _, row in lap_rows.iterrows():
        driver = row['Driver']
        personal_fastest = fastest_so_far.get(driver, pd.NaT)
        if pd.notna(row['Position']):
            gap_leader = row['Time'] - leader_time
            gap_front = (row['Time'] - prev_time) if prev_time is not None else pd.Timedelta(0)
            prev_time = row['Time']
            pos_display = int(row['Position'])
        else:
            gap_leader, gap_front, pos_display = pd.NaT, pd.NaT, '-'
        records.append({
            'Pos': pos_display,
            'Driver': driver,
            'Tyre': row['Compound'] if pd.notna(row['Compound']) else '-',
            'Gap to Leader': _format_gap(gap_leader),
            'Gap to Front': _format_gap(gap_front),
            'Lap Time': _format_laptime(row['LapTime']),
            'Fastest Lap': _format_laptime(personal_fastest),
            '_is_fastest': pd.notna(overall_fastest) and personal_fastest == overall_fastest,
        })

    df = pd.DataFrame(records)
    fastest_mask = df['_is_fastest'].tolist()
    return df.drop(columns='_is_fastest'), fastest_mask


@st.fragment
def render_race_status_table(session, year, race, session_type_label, session_type_code):
    # @st.fragment so moving the lap selector only re-runs this table, not the whole
    # page (which would otherwise also re-render the track map/telemetry dashboard).
    if session_type_code not in ('R', 'S'):
        st.caption("Race status is only meaningful for Race/Sprint sessions.")
        return

    st.markdown("### Race Status")
    max_lap = int(session.laps['LapNumber'].max())
    lap_number = st.number_input(
        "Lap", min_value=1, max_value=max_lap, value=max_lap, key=f"status_lap_{year}_{race}_{session_type_code}")

    try:
        df, fastest_mask = get_lap_status_table(year, race, session_type_code, int(lap_number))
    except Exception as e:
        st.warning(f"Error building race status table: {e}")
        return

    if df.empty:
        st.caption(f"No data available for lap {lap_number}.")
        return

    compound_colors = {c: get_compound_color_cached(year, race, session_type_code, c)
                        for c in df['Tyre'].unique() if c != '-'}

    def _highlight(row):
        styles = [''] * len(row)
        if fastest_mask[row.name]:
            styles = ['background-color: #4B0082; color: white'] * len(row)
        tyre_col = row.index.get_loc('Tyre')
        color = compound_colors.get(row['Tyre'])
        if color:
            styles[tyre_col] = f'background-color: {color}; color: black'
        return styles

    st.dataframe(
        df.style.apply(_highlight, axis=1),
        hide_index=True,
        use_container_width=True,
        # Streamlit's default dataframe height only fits ~10 rows before scrolling
        # internally - sized to the actual row count instead, so all ~20-22 drivers
        # on the grid are visible at once without scrolling inside the table.
        height=(len(df) + 1) * 35 + 3,
    )
    st.caption("Purple row = fastest lap of the race so far. Gaps are estimated from cumulative "
               "session time at the same lap number, so they're approximate once a car has been lapped.")


@st.cache_data(show_spinner="Building track-limit corridor...", persist="disk")
def get_track_limit_envelope(year, race, session_type, bin_size=TRACK_LIMIT_BIN_SIZE_M, top_n=None):
    """Approximate legal-racing corridor: the min/max lateral offset (from the overall
    fastest lap's line) covered by clean fastest laps. This is NOT the physical white-
    line/curb geometry FastF1 doesn't expose that - it's the spread of racing lines
    actually used at racing pace, which is what we use as the "normal" corridor to flag
    a driver running wider than the field on a given lap.

    top_n: if set, only the N fastest of those clean laps (by lap time, not finishing
    position - a driver can finish well on strategy with a slow lap) go into the
    corridor, so it represents "how the quick laps actually used the track" rather than
    being diluted by the whole field including cars managing tyres or running
    defensively. None (default) uses every driver's fastest clean lap, as before.
    """
    session = get_session_data(year, race, session_type)
    clean_laps = session.laps.pick_wo_box().pick_quicklaps()
    fastest_per_driver = clean_laps.loc[clean_laps.groupby('Driver')['LapTime'].idxmin()]
    if top_n is not None:
        fastest_per_driver = fastest_per_driver.sort_values('LapTime').head(top_n)

    ref_lap = session.laps.pick_fastest()
    ref_tel = _drop_stuck_position_samples(ref_lap.get_telemetry().reset_index(drop=True))
    cd_raw = ref_tel['Distance'].to_numpy()
    cx_raw = ref_tel['X'].to_numpy()
    cy_raw = ref_tel['Y'].to_numpy()

    bins = np.arange(0, cd_raw.max(), bin_size)
    cx = np.interp(bins, cd_raw, cx_raw)
    cy = np.interp(bins, cd_raw, cy_raw)

    # The track is a closed loop (bin 0 and the last bin both sit at the start/finish
    # line), so smooth and differentiate the centerline as periodic (np.roll) rather
    # than as an open-ended array. A raw two-point gradient at 5m spacing is noisy
    # enough that tiny heading wobbles flip the normal direction bin-to-bin, which
    # turns small along-track misalignment into large spurious lateral offsets
    # (visible as spikes/over-wide sections in the corridor) - smoothing first, and
    # wrapping at the seam, removes both that jitter and the start/finish edge artifact.
    smooth_window = max(3, int(round(25.0 / bin_size)) | 1)  # ~25m, forced odd
    pad = smooth_window // 2
    kernel = np.ones(smooth_window) / smooth_window

    def _smooth_periodic(arr):
        padded = np.concatenate([arr[-pad:], arr, arr[:pad]])
        return np.convolve(padded, kernel, mode='valid')

    cx_smooth, cy_smooth = _smooth_periodic(cx), _smooth_periodic(cy)
    tx = np.roll(cx_smooth, -1) - np.roll(cx_smooth, 1)
    ty = np.roll(cy_smooth, -1) - np.roll(cy_smooth, 1)
    tnorm = np.sqrt(tx ** 2 + ty ** 2)
    tnorm[tnorm == 0] = 1
    tx, ty = tx / tnorm, ty / tnorm
    nx, ny = -ty, tx

    # Per-driver extreme offset per bin, not a single global min/max: a true min/max
    # across the whole field means one driver's one-off wide moment through a single
    # corner (e.g. a defensive line or a slow-corner scruffy exit) drags the entire
    # corridor out at that bin, which is exactly what showed up as isolated width
    # spikes/bulges when checked against real data (verified on 2024 Austria Qualifying:
    # one driver at -67m offset through Turn 9 vs the rest of the field at -20 to -30m).
    driver_list = list(fastest_per_driver['Driver'])
    driver_col = {drv: i for i, drv in enumerate(driver_list)}
    n_drivers = len(driver_list)

    if n_drivers == 0:
        # No driver has a clean quicklap this session (wet/red-flagged/very short
        # session) - return explicitly invalid rather than silently computing an
        # all-NaN envelope, which would make every offset comparison downstream
        # evaluate false and report a false-clean "zero violations" result.
        return {
            'bins': bins, 'cx': cx, 'cy': cy, 'nx': nx, 'ny': ny,
            'min_off': np.full(len(bins), np.nan), 'max_off': np.full(len(bins), np.nan),
            'bin_size': bin_size, 'driver_count': 0, 'valid': False,
        }

    # Interpolate each driver's own continuous offset profile onto the bin grid,
    # rather than binning raw samples into the nearest 5m bucket. Checked against real
    # data (2026 Austria): at racing speed, telemetry samples land ~7m apart on
    # average and up to 90m apart on the fastest straights, so nearest-bin binning left
    # roughly HALF of all bins with zero raw samples from any given driver. Those gaps
    # used to be forward/back-filled from whichever bin did have data, which produced a
    # visible step exactly at the fill boundary - the "spikes" on straights (and, with
    # fewer bins covered per corner too, some in corners). Interpolating gives every
    # bin a legitimate value with no fill-boundary artifacts, and needs no bin-count
    # threshold to kick in - it's correct at any speed/bin size.
    offset_by_driver = np.full((len(bins), n_drivers), np.nan)

    # Fetched via the shared per-lap cache (get_single_lap_telemetry) rather than each
    # driver's fastest lap being fetched inline here - a driver whose fastest lap is
    # already cached from elsewhere (e.g. they're also one of the currently-selected
    # drivers) costs nothing to include in the corridor.
    lap_number_by_driver = dict(zip(fastest_per_driver['Driver'],
                                     fastest_per_driver['LapNumber'].astype(int)))
    fetch_args = [(year, race, session_type, drv, lap_number_by_driver[drv]) for drv in driver_list]
    for drv, tel in zip(driver_list, _fetch_laps(fetch_args)):
        if tel.empty:
            continue
        px, py, pdist = tel['X'].to_numpy(), tel['Y'].to_numpy(), tel['Distance'].to_numpy()
        order = np.argsort(pdist)
        pdist_sorted, px_sorted, py_sorted = pdist[order], px[order], py[order]
        idx = np.clip(np.round(pdist_sorted / bin_size).astype(int), 0, len(bins) - 1)
        raw_offset = (px_sorted - cx[idx]) * nx[idx] + (py_sorted - cy[idx]) * ny[idx]
        col = driver_col[drv]
        # np.interp clips to the boundary value outside pdist's own range (e.g. if this
        # driver's lap measured slightly shorter/longer than the reference lap), rather
        # than extrapolating wildly - a safe, bounded fallback for that edge.
        offset_by_driver[:, col] = np.interp(bins, pdist_sorted, raw_offset)

    # Trim roughly the single widest driver on each side per bin (a percentile scaled
    # to the field size) instead of taking the absolute extreme, so the corridor
    # reflects the field's normal spread rather than one outlier lap.
    trim_pct = 100.0 / n_drivers if n_drivers else 5.0
    # A bin outside every single driver's own individually-measured lap range (rare -
    # cars don't all record exactly the same total lap distance) leaves that bin
    # all-NaN pre-trim; nanpercentile handles it (produces NaN, patched below by the
    # ffill/bfill) but warns every time, which is just log noise for an already-
    # handled case.
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        min_off = np.nanpercentile(offset_by_driver, trim_pct, axis=1)
        max_off = np.nanpercentile(offset_by_driver, 100.0 - trim_pct, axis=1)

    # Defensive only at this point (interpolation covers every bin already) - guards
    # the pathological case of a bin outside every single driver's own lap range.
    min_off = pd.Series(min_off).ffill().bfill().to_numpy()
    max_off = pd.Series(max_off).ffill().bfill().to_numpy()

    return {
        'bins': bins, 'cx': cx, 'cy': cy, 'nx': nx, 'ny': ny,
        'min_off': min_off, 'max_off': max_off, 'bin_size': bin_size,
        'driver_count': len(fastest_per_driver), 'valid': True,
    }


@st.cache_data(show_spinner=False, persist="disk")
def get_driver_track_limit_violations(year, race, session_type, driver, lap_numbers):
    telemetry_df = get_driver_lap_telemetry(year, race, session_type, driver, lap_numbers)
    if telemetry_df.empty:
        return telemetry_df.assign(Excess=[])

    envelope = get_track_limit_envelope(year, race, session_type)
    if not envelope.get('valid', True):
        # No corridor data this session - explicitly empty (no violations found) is
        # not the right answer here, but there's no "unknown" DataFrame state, so an
        # empty result plus the caller checking envelope['valid'] itself for messaging
        # is how this stays distinguishable from a genuine zero-violations result.
        return telemetry_df.iloc[0:0].copy().assign(Excess=[])
    bin_size = envelope['bin_size']
    idx = np.clip(np.round(telemetry_df['Distance'].to_numpy() / bin_size).astype(int),
                  0, len(envelope['bins']) - 1)
    px, py = telemetry_df['X'].to_numpy(), telemetry_df['Y'].to_numpy()
    offset = (px - envelope['cx'][idx]) * envelope['nx'][idx] + (py - envelope['cy'][idx]) * envelope['ny'][idx]

    over = offset - envelope['max_off'][idx]
    under = envelope['min_off'][idx] - offset
    excess = np.maximum(over, under)
    excess = np.where(excess > 0, excess, 0)

    violations = telemetry_df[excess > 0].copy()
    violations['Excess'] = excess[excess > 0]
    return violations


def get_tangent_and_pos(ref_tel, session_time):
    idx = (ref_tel['SessionTime'] - session_time).abs().argmin()
    x, y = ref_tel['X'].iloc[idx], ref_tel['Y'].iloc[idx]

    idx_prev = (idx - 2) % len(ref_tel)
    idx_next = (idx + 2) % len(ref_tel)

    dx = ref_tel['X'].iloc[idx_next] - ref_tel['X'].iloc[idx_prev]
    dy = ref_tel['Y'].iloc[idx_next] - ref_tel['Y'].iloc[idx_prev]

    norm = (dx ** 2 + dy ** 2) ** 0.5
    if norm == 0:
        norm = 1
    return x, y, dx / norm, dy / norm


def add_sector_gate(fig, ref_tel, session_time, color, label):
    x, y, tx, ty = get_tangent_and_pos(ref_tel, session_time)
    nx, ny = -ty, tx
    gate_width = 250
    fig.add_trace(go.Scattergl(
        x=[x - nx * gate_width, x + nx * gate_width],
        y=[y - ny * gate_width, y + ny * gate_width],
        mode='lines', line=dict(color=color, width=4),
        name=label, hoverinfo='skip',
    ))


HOVER_SPEED_DISTANCE = (
    'Lap %{customdata[2]}<br>Speed: %{customdata[0]:.0f} km/h<br>Distance: %{customdata[1]:.0f} m<extra></extra>'
)


def _speed_distance_customdata(tel_df, lap_number):
    return np.stack([tel_df['Speed'], tel_df['Distance'], [lap_number] * len(tel_df)], axis=-1)


def _driver_hover_template(driver):
    return f'{driver} ' + HOVER_SPEED_DISTANCE


def _apply_distance_or_corner_xaxis(fig_obj, corner_tick_vals, corner_tick_text, row, col):
    if corner_tick_vals:
        fig_obj.update_xaxes(title_text='Corner', tickmode='array', tickvals=corner_tick_vals,
                              ticktext=corner_tick_text, color='white', gridcolor=TRACK_COLOR, row=row, col=col)
    else:
        fig_obj.update_xaxes(title_text='Distance (m)', color='white', gridcolor=TRACK_COLOR, row=row, col=col)


def _style_subplot_figure(fig_obj, height):
    fig_obj.update_layout(
        plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
        legend=dict(font=dict(color='white'), bgcolor=BG_COLOR),
        margin=dict(l=10, r=10, t=40, b=10), height=height,
        font=dict(color='white'),
    )


def _braking_points(lap_tel):
    """X,Y of each point where the Brake channel rises from off to on along one lap
    (sorted by Distance first) - i.e. where the driver first gets on the brakes for
    each corner, not every sample where the brake happens to be applied.
    """
    tel = lap_tel.sort_values('Distance')
    brake = tel['Brake'].astype(bool).to_numpy()
    if len(brake) < 2:
        return np.array([]), np.array([])
    rising = np.zeros(len(brake), dtype=bool)
    rising[1:] = (~brake[:-1]) & brake[1:]
    x, y = tel['X'].to_numpy(), tel['Y'].to_numpy()
    return x[rising], y[rising]


with st.spinner("Downloading and processing session telemetry..."):
    try:
        session = get_session_data(selected_year, selected_race, session_dict[selected_session_type])
    except Exception as e:
        st.error(f"Couldn't load {selected_race} {selected_year} ({selected_session_type}): {e}")
        st.stop()


@st.fragment
def render_driver_dashboard(session):
    # --- 4. Driver & View Mode Settings ---
    # This whole function is an @st.fragment: Streamlit scopes reruns triggered by any
    # widget inside it (driver, view mode, lap slider/interval) to just this function,
    # instead of re-running the whole script and redrawing the whole page.
    st.markdown("---")
    drivers = session.drivers
    driver_names = [session.get_driver(d)['Abbreviation'] for d in drivers]

    col_d, col_v = st.columns(2)
    with col_d:
        default_driver = ['VER'] if 'VER' in driver_names else driver_names[:1]
        selected_drivers = st.multiselect("Select Driver(s)", driver_names, default=default_driver)

    with col_v:
        view_mode = st.selectbox("View Mode", ["Single Lap", "Average Trend"])

    if not selected_drivers:
        st.warning("Select at least one driver.")
        st.stop()

    # --- Track map display options ---
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        corridor_all_drivers = st.checkbox(
            "Corridor: use all drivers", value=False,
            help="Off (default): build the track-limit corridor from only the fastest N drivers' "
                 "laps (ranked by lap time, not finishing position), so it reflects racing-pace "
                 "lines rather than being diluted by the whole field. On: every driver's fastest "
                 "clean lap, like before.")
        corridor_top_n = None
        if not corridor_all_drivers:
            corridor_top_n = st.slider("Corridor: fastest N laps", min_value=3,
                                        max_value=max(len(driver_names), 3),
                                        value=min(8, len(driver_names)))
    with col_opt2:
        color_by_speed = False
        if view_mode == "Single Lap" and len(selected_drivers) == 1:
            color_by_speed = st.checkbox(
                "Color racing line by speed", value=False,
                help="Single driver/lap only - replaces the driver-colored line with a speed "
                     "heatmap along the racing line (braking zones cold, straights hot).")
    with col_opt3:
        show_braking_points = st.checkbox(
            "Show braking points", value=(view_mode == "Single Lap"),
            help="Marks where each driver first gets on the brakes for each corner "
                 "(the Brake channel's rising edge), not every braking sample.")

    # Filter out slow laps (in/out laps, safety cars) so they don't ruin the visualization.
    # Fetched per-driver, so re-selecting a driver already viewed this session reuses
    # Streamlit's cache instead of re-querying FastF1.
    driver_laps_map = {d: session.laps.pick_driver(d).pick_quicklaps() for d in selected_drivers}
    driver_laps_map = {d: laps for d, laps in driver_laps_map.items() if not laps.empty}

    if not driver_laps_map:
        st.warning("No valid quick laps found for the selected driver(s) in this session.")
        st.stop()
    if len(driver_laps_map) < len(selected_drivers):
        missing = [d for d in selected_drivers if d not in driver_laps_map]
        st.caption(f"No valid quick laps for: {', '.join(missing)} - excluded below.")

    min_lap = min(int(laps['LapNumber'].min()) for laps in driver_laps_map.values())
    max_lap = max(int(laps['LapNumber'].max()) for laps in driver_laps_map.values())

    if view_mode == "Single Lap":
        if color_by_speed and len(driver_laps_map) == 1:
            # Speed-heatmap mode needs one specific lap picked server-side (its
            # marker colors are computed from that lap's own Speed values), so it
            # keeps the plain slider + fragment rerun rather than the client-side
            # scrubber used below.
            selected_lap = st.slider("Select Lap", min_value=min_lap, max_value=max_lap, value=min_lap)
            laps_to_plot_map = {d: laps[laps['LapNumber'] == selected_lap] for d, laps in driver_laps_map.items()}
        else:
            # Every quick lap is fetched once, up front (the same call "Average
            # Trend" mode already uses) - the track map below then lets you scrub
            # between laps entirely client-side (a Plotly slider, no Streamlit
            # rerun and no repeat FastF1 fetch per lap), which is what actually
            # made "Select Lap" feel slow before: every lap change forced a full
            # fragment rerun plus a fresh per-lap telemetry fetch.
            laps_to_plot_map = driver_laps_map
            st.caption("Drag the lap slider under the track map to switch laps instantly - "
                       "no page reload needed.")
        alpha_val = 1.0
        line_width = 2

    else:  # Average Trend
        laps_to_plot_map = driver_laps_map
        alpha_val = 0.15
        line_width = 1.5

    # Downsample display-only traces (racing line, throttle/brake) once enough laps/
    # drivers are on screen at once - corridor math and violation detection upstream
    # still run on full-resolution telemetry, this only thins what gets drawn/sent to
    # the browser, which is what actually costs time with many overlapping laps.
    total_lap_count = sum(len(v) for v in laps_to_plot_map.values())
    if total_lap_count <= 3:
        decimate_stride = 1
    elif total_lap_count <= 12:
        decimate_stride = 2
    elif total_lap_count <= 30:
        decimate_stride = 3
    else:
        decimate_stride = 5

    def _decimate(df):
        return df.iloc[::decimate_stride] if decimate_stride > 1 and len(df) > decimate_stride else df

    # --- 5. Rendering the Dashboard ---
    with st.spinner(f"Plotting trajectories for {', '.join(laps_to_plot_map.keys())}..."):
        # --- SHARED DATA PREP --- everything below needs this; if it fails, nothing
        # else can render, so this is the one place a failure legitimately stops the
        # whole dashboard rather than just one panel.
        try:
            driver_colors, driver_dash, seen_colors = {}, {}, {}
            for d in laps_to_plot_map:
                color = get_driver_color_cached(
                    selected_year, selected_race, session_dict[selected_session_type], d)
                driver_colors[d] = color
                driver_dash[d] = DASH_OPTIONS[seen_colors.get(color, 0) % len(DASH_OPTIONS)]
                seen_colors[color] = seen_colors.get(color, 0) + 1

            ref_tel, sf_time, sector1_time, sector2_time, glitch_fraction = get_reference_lap_telemetry(
                selected_year, selected_race, session_dict[selected_session_type])

            if glitch_fraction > 0.40:
                st.error(
                    f"Position telemetry for this session looks severely degraded "
                    f"({glitch_fraction:.0%} of the fastest lap's samples failed a physical-plausibility "
                    f"check) - this is a live-timing data quality issue, not something filtering can "
                    f"fix. The track outline, corridor, and racing lines below are unreliable."
                )
            elif glitch_fraction > 0.15:
                st.warning(
                    f"Position telemetry for this session looks partially degraded "
                    f"({glitch_fraction:.0%} of the fastest lap's samples failed a physical-plausibility "
                    f"check). The track outline/corridor below may have some rough spots."
                )

            envelope = get_track_limit_envelope(
                selected_year, selected_race, session_dict[selected_session_type], top_n=corridor_top_n)

            plot_lap_numbers_map = {
                d: tuple(sorted(laps['LapNumber'].astype(int).tolist()))
                for d, laps in laps_to_plot_map.items() if not laps.empty
            }
            telemetry_map = {
                d: get_driver_lap_telemetry(
                    selected_year, selected_race, session_dict[selected_session_type], d, lap_numbers)
                for d, lap_numbers in plot_lap_numbers_map.items()
            }

            # Corner positions/labels, computed once here since the track map and both
            # throttle/brake plots below all need them.
            corners = get_circuit_corners(
                selected_year, selected_race, session_dict[selected_session_type])
            corner_offset_x, corner_offset_y, corner_labels = None, None, None
            corner_tick_vals, corner_tick_text = None, None
            if corners is not None:
                # Scale the label offset to the track's own footprint rather than a fixed
                # number: a flat 500-unit offset put labels right on top of the track line
                # for larger circuits (500 units is a much smaller fraction of a big track's
                # span than a small one), which is why they looked cramped/underneath it.
                track_span = max(ref_tel['X'].max() - ref_tel['X'].min(),
                                  ref_tel['Y'].max() - ref_tel['Y'].min())
                offset_dist = max(track_span * 0.08, 400)
                # corners['Angle'] is FastF1's own precomputed direction for exactly this
                # offset (see their circuit-info example): rotating the vector (offset, 0)
                # by this angle points away from the track at that corner. We had sin/cos
                # swapped between X and Y, which rotates that vector 90 degrees off from
                # the intended direction - correct for many corners by chance, wrong for
                # just as many others, which is why labels still sat on the track edge
                # even after enlarging the offset distance alone.
                angle_rad = np.radians(corners['Angle'])
                corner_offset_x = corners['X'] + offset_dist * np.cos(angle_rad)
                corner_offset_y = corners['Y'] + offset_dist * np.sin(angle_rad)
                corner_labels = [
                    f"{int(row['Number'])}{row['Letter'] if pd.notna(row['Letter']) else ''}"
                    for _, row in corners.iterrows()
                ]

                # Distance-along-lap nearest each corner (via the reference lap's own
                # Distance channel), so the throttle/brake plots can label their x-axis
                # by corner instead of raw metres.
                ref_x, ref_y, ref_d = ref_tel['X'].to_numpy(), ref_tel['Y'].to_numpy(), ref_tel['Distance'].to_numpy()
                corner_dists = [
                    ref_d[np.argmin((ref_x - row['X']) ** 2 + (ref_y - row['Y']) ** 2)]
                    for _, row in corners.iterrows()
                ]
                order = np.argsort(corner_dists)
                corner_tick_vals = [corner_dists[i] for i in order]
                corner_tick_text = [corner_labels[i] for i in order]

            # Single Lap mode (color-by-speed excepted, handled in its own branch
            # below) now always loads every quick lap up front - see the view-mode
            # block above - so both the track map and throttle/brake panels can
            # scrub between laps via a client-side Plotly frame/slider instead of
            # a plain per-lap-fetching Streamlit slider.
            is_single_lap_scrub = (view_mode == "Single Lap"
                                    and not (color_by_speed and len(telemetry_map) == 1))
        except Exception as e:
            st.warning(f"Error loading data for {', '.join(laps_to_plot_map.keys())}: {e}")
            st.stop()

        # --- TRACK MAP --- its own try/except so a failure here (e.g. a bad corner
        # entry) can't take down the throttle/brake or tyre panels below, and vice versa.
        try:
            fig = go.Figure()

            # --- BASE TRACK LAYER ---
            fig.add_trace(go.Scattergl(
                x=ref_tel['X'], y=ref_tel['Y'], mode='lines',
                line=dict(color=TRACK_COLOR, width=12),
                name='Circuit Layout', hoverinfo='skip',
            ))

            # --- TRACK LIMIT CORRIDOR ---
            # Shaded band = spread of every driver's own fastest lap around the reference
            # line. It approximates the "normal" racing corridor, not the physical curbs.
            # Skipped (with an explicit note, not a silent empty corridor) if no driver
            # had a clean quicklap this session to build one from.
            if envelope['valid']:
                edge_x = envelope['cx'] + envelope['nx'] * envelope['max_off']
                edge_y = envelope['cy'] + envelope['ny'] * envelope['max_off']
                other_edge_x = envelope['cx'] + envelope['nx'] * envelope['min_off']
                other_edge_y = envelope['cy'] + envelope['ny'] * envelope['min_off']

                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode='lines',
                    line=dict(color=CORRIDOR_COLOR, width=1), hoverinfo='skip', showlegend=False,
                ))
                corridor_label = (f"Track-limit corridor ({envelope['driver_count']} drivers' fastest laps)"
                                   if corridor_all_drivers else
                                   f"Track-limit corridor (fastest {envelope['driver_count']} laps)")
                fig.add_trace(go.Scatter(
                    x=other_edge_x, y=other_edge_y, mode='lines',
                    line=dict(color=CORRIDOR_COLOR, width=1), fill='tonexty',
                    fillcolor='rgba(58, 58, 85, 0.35)', hoverinfo='skip',
                    name=corridor_label,
                ))
            else:
                st.info("No driver had a clean quick lap this session, so no track-limit corridor "
                        "could be built - the map below shows racing lines only, with no corridor "
                        "shading or 'wide of corridor' markers.")

            braking_points_by_driver = {d: ([], []) for d in telemetry_map}

            if is_single_lap_scrub:
                # --- SINGLE LAP: CLIENT-SIDE SCRUBBER ---
                # One persistent trace per driver (line, plus braking-point/violation
                # marker traces if enabled) rather than one trace per lap - laps are
                # switched by swapping each trace's data via a Plotly frame, showing
                # only the single active lap (no faded multi-lap context) to match
                # this mode's look.
                all_lap_numbers = sorted(set(n for nums in plot_lap_numbers_map.values() for n in nums))
                if not all_lap_numbers:
                    st.warning("No laps available to plot.")
                else:
                    violations_by_driver = {}
                    if envelope['valid']:
                        for d, lap_numbers in plot_lap_numbers_map.items():
                            violations_by_driver[d] = get_driver_track_limit_violations(
                                selected_year, selected_race, session_dict[selected_session_type],
                                d, lap_numbers)

                    line_trace_idx, brake_trace_idx, violation_trace_idx = {}, {}, {}

                    for d, lap_numbers in plot_lap_numbers_map.items():
                        first_lap = lap_numbers[0]
                        first_lap_tel = telemetry_map[d][telemetry_map[d]['LapNumber'] == first_lap]
                        first_tel = _decimate(first_lap_tel)

                        line_trace_idx[d] = len(fig.data)
                        fig.add_trace(go.Scattergl(
                            x=first_tel['X'], y=first_tel['Y'], mode='lines',
                            line=dict(color=driver_colors[d], width=line_width, dash=driver_dash[d]),
                            name=f'{d} Trajectory', legendgroup=f'driver_{d}',
                            customdata=_speed_distance_customdata(first_tel, first_lap),
                            hovertemplate=_driver_hover_template(d),
                        ))

                        if show_braking_points:
                            bx, by = _braking_points(first_lap_tel)
                            brake_trace_idx[d] = len(fig.data)
                            fig.add_trace(go.Scattergl(
                                x=bx, y=by, mode='markers',
                                marker=dict(symbol='triangle-down', size=9, color=driver_colors[d],
                                            line=dict(color='white', width=1)),
                                name=f'{d} braking points', legendgroup=f'driver_{d}', hoverinfo='skip',
                            ))

                        if envelope['valid']:
                            v = violations_by_driver[d]
                            v_first = v[v['LapNumber'] == first_lap] if not v.empty else v
                            violation_trace_idx[d] = len(fig.data)
                            is_first_violation_trace = len(violation_trace_idx) == 1
                            fig.add_trace(go.Scattergl(
                                x=v_first['X'], y=v_first['Y'], mode='markers',
                                marker=dict(size=5, color=VIOLATION_COLOR, symbol='circle'),
                                name='Wide of corridor' if is_first_violation_trace else None,
                                legendgroup='violations', showlegend=is_first_violation_trace,
                                customdata=(np.stack([v_first['LapNumber'], v_first['Excess']], axis=-1)
                                            if not v_first.empty else np.empty((0, 2))),
                                hovertemplate=(f'{d} Lap %{{customdata[0]}}<br>+%{{customdata[1]:.1f}} '
                                               'm wide<extra></extra>'),
                            ))

                    frames = []
                    for lap_number in all_lap_numbers:
                        frame_data, frame_traces = [], []
                        for d, lap_numbers in plot_lap_numbers_map.items():
                            has_lap = lap_number in lap_numbers
                            lap_tel = (telemetry_map[d][telemetry_map[d]['LapNumber'] == lap_number]
                                       if has_lap else telemetry_map[d].iloc[0:0])

                            frame_traces.append(line_trace_idx[d])
                            if has_lap:
                                plot_tel = _decimate(lap_tel)
                                frame_data.append(go.Scattergl(
                                    x=plot_tel['X'], y=plot_tel['Y'],
                                    customdata=_speed_distance_customdata(plot_tel, lap_number),
                                ))
                            else:
                                frame_data.append(go.Scattergl(x=[], y=[], customdata=np.empty((0, 3))))

                            if show_braking_points:
                                frame_traces.append(brake_trace_idx[d])
                                bx, by = _braking_points(lap_tel) if has_lap else (np.array([]), np.array([]))
                                frame_data.append(go.Scattergl(x=bx, y=by))

                            if envelope['valid']:
                                frame_traces.append(violation_trace_idx[d])
                                v = violations_by_driver[d]
                                v_lap = v[v['LapNumber'] == lap_number] if (has_lap and not v.empty) else v.iloc[0:0]
                                frame_data.append(go.Scattergl(
                                    x=v_lap['X'], y=v_lap['Y'],
                                    customdata=(np.stack([v_lap['LapNumber'], v_lap['Excess']], axis=-1)
                                                if not v_lap.empty else np.empty((0, 2))),
                                ))
                        frames.append(go.Frame(name=str(lap_number), data=frame_data, traces=frame_traces))
                    fig.frames = frames

                    fig.update_layout(sliders=[dict(
                        active=0,
                        currentvalue=dict(prefix='Lap: ', font=dict(color='white')),
                        pad=dict(t=20), font=dict(color='white'),
                        steps=[dict(
                            method='animate',
                            args=[[str(lap_number)], dict(mode='immediate',
                                                           frame=dict(duration=0, redraw=True),
                                                           transition=dict(duration=0))],
                            label=str(lap_number),
                        ) for lap_number in all_lap_numbers],
                    )])

                    if envelope['valid']:
                        laps_with_violations = {
                            d: sorted(v['LapNumber'].unique().astype(int).tolist())
                            for d, v in violations_by_driver.items() if not v.empty
                        }
                        if laps_with_violations:
                            summary = "; ".join(f"{d}: laps {', '.join(map(str, laps))}"
                                                 for d, laps in laps_with_violations.items())
                            st.caption(f"Ran wide of the fastest-lap corridor on - {summary} "
                                       "(scrub to those laps to see exactly where).")
                        else:
                            st.caption("All selected drivers stayed within the fastest-lap corridor "
                                       "across every loaded lap.")
                    else:
                        st.caption("Track-limit violations unavailable - no corridor data this session.")

            else:
                # --- MAIN RACING LINE(S) ---
                if color_by_speed and len(telemetry_map) == 1:
                    # Single driver/lap only (enforced by the checkbox's own visibility
                    # condition above): a speed heatmap in place of the normal driver-
                    # colored line - braking zones read cold, straights hot.
                    d = next(iter(telemetry_map))
                    lap_tel = telemetry_map[d]
                    plot_tel = _decimate(lap_tel)
                    fig.add_trace(go.Scattergl(
                        x=plot_tel['X'], y=plot_tel['Y'], mode='markers',
                        marker=dict(
                            size=4, color=plot_tel['Speed'], colorscale='Turbo', showscale=True,
                            colorbar=dict(title=dict(text='Speed (km/h)', font=dict(color='white')),
                                          tickfont=dict(color='white'), x=1.02),
                        ),
                        name=f'{d} Trajectory (speed)',
                        customdata=_speed_distance_customdata(plot_tel, plot_tel['LapNumber'].iloc[0]),
                        hovertemplate=_driver_hover_template(d),
                    ))
                    if show_braking_points:
                        bx, by = _braking_points(lap_tel)
                        braking_points_by_driver[d] = (list(bx), list(by))
                else:
                    # Average Trend mode: one averaged racing line per driver (mean
                    # X/Y/Speed at each distance-along-lap bin, across every quick lap -
                    # see get_average_lap_trend), always colored by average speed
                    # rather than a flat driver color, since this is meant to read as
                    # a single representative "trend" lap, not a per-lap overlay.
                    avg_profiles = {
                        d: get_average_lap_trend(
                            selected_year, selected_race, session_dict[selected_session_type], d)
                        for d in telemetry_map
                    }
                    avg_profiles = {d: p for d, p in avg_profiles.items() if not p.empty}
                    if not avg_profiles:
                        st.warning("No average trend could be computed for the selected driver(s).")
                    else:
                        # Shared cmin/cmax across every driver's trace so the one visible
                        # colorbar applies consistently to all of them, not just whichever
                        # trace happens to own it.
                        all_speeds = np.concatenate([p['Speed'].to_numpy() for p in avg_profiles.values()])
                        cmin, cmax = float(all_speeds.min()), float(all_speeds.max())
                        for i, (d, profile) in enumerate(avg_profiles.items()):
                            fig.add_trace(go.Scattergl(
                                x=profile['X'], y=profile['Y'], mode='markers',
                                marker=dict(
                                    size=6, color=profile['Speed'], colorscale='Turbo',
                                    cmin=cmin, cmax=cmax, showscale=(i == 0),
                                    colorbar=dict(title=dict(text='Avg speed (km/h)', font=dict(color='white')),
                                                  tickfont=dict(color='white'), x=1.02),
                                ),
                                name=f'{d} Average Trend',
                                customdata=np.stack([profile['Speed'], profile['Distance']], axis=-1),
                                hovertemplate=(f'{d}<br>Avg speed: %{{customdata[0]:.0f}} km/h<br>'
                                               'Distance: %{customdata[1]:.0f} m<extra></extra>'),
                            ))

                    # Braking points, if enabled, are still drawn from every raw quick
                    # lap (not the averaged line) - averaging would blur exactly where
                    # they actually happen.
                    if show_braking_points:
                        for d, telemetry_df in telemetry_map.items():
                            for _, lap_tel in telemetry_df.groupby('LapNumber'):
                                bx, by = _braking_points(lap_tel)
                                braking_points_by_driver[d][0].extend(bx)
                                braking_points_by_driver[d][1].extend(by)

                # --- BRAKING POINTS (per driver) ---
                # Where the Brake channel first rises (not every braking sample), i.e. the
                # point a driver gets on the brakes for each corner.
                if show_braking_points:
                    for d, (bx, by) in braking_points_by_driver.items():
                        if len(bx):
                            fig.add_trace(go.Scattergl(
                                x=bx, y=by, mode='markers',
                                marker=dict(symbol='triangle-down', size=9, color=driver_colors[d],
                                            line=dict(color='white', width=1)),
                                name=f'{d} braking points', legendgroup=f'driver_{d}',
                                hoverinfo='skip',
                            ))

                # --- TRACK LIMIT VIOLATIONS (per driver) ---
                # Points where a selected driver ran wider than every other driver's own
                # fastest lap at that point on track. Unavailable (explicitly, not as a
                # silent "zero violations") when there's no corridor to compare against.
                if not envelope['valid']:
                    st.caption("Track-limit violations unavailable - no corridor data this session.")
                else:
                    violation_summaries = []
                    for d, lap_numbers in plot_lap_numbers_map.items():
                        violations = get_driver_track_limit_violations(
                            selected_year, selected_race, session_dict[selected_session_type], d, lap_numbers)
                        if violations.empty:
                            continue
                        fig.add_trace(go.Scattergl(
                            x=violations['X'], y=violations['Y'], mode='markers',
                            marker=dict(size=5, color=VIOLATION_COLOR, symbol='circle'),
                            name='Wide of corridor' if not violation_summaries else None,
                            legendgroup='violations', showlegend=(not violation_summaries),
                            customdata=np.stack([violations['LapNumber'], violations['Excess']], axis=-1),
                            hovertemplate=f'{d} Lap %{{customdata[0]}}<br>+%{{customdata[1]:.1f}} m wide<extra></extra>',
                        ))
                        violation_summaries.append(
                            f"{d}: {len(violations)} points (max {violations['Excess'].max():.1f} m beyond)")

                    if violation_summaries:
                        st.caption("Ran wider than the fastest-lap corridor - " + "; ".join(violation_summaries))
                    else:
                        st.caption(
                            "All selected drivers stayed within the fastest-lap corridor for the selected lap(s).")

            # --- SECTOR GATES ---
            # Guarded against NaT: a fastest lap occasionally has a missing sector
            # timestamp (a real FastF1 data gap), which previously crashed sector-gate
            # placement and, via one blanket except around the whole dashboard, took
            # every other panel down with it too. Now a missing sector time just skips
            # that one gate.
            for session_time, color, label in (
                (sf_time, SECTOR_COLORS['Start/Finish'], 'Start/Finish'),
                (sector1_time, SECTOR_COLORS['Sector 1 Split'], 'Sector 1 Split'),
                (sector2_time, SECTOR_COLORS['Sector 2 Split'], 'Sector 2 Split'),
            ):
                if pd.notna(session_time):
                    add_sector_gate(fig, ref_tel, session_time, color, label)

            # --- TURN NUMBERS ---
            if corners is not None:
                for x0, y0, x1, y1 in zip(corners['X'], corners['Y'], corner_offset_x, corner_offset_y):
                    fig.add_trace(go.Scattergl(
                        x=[x0, x1], y=[y0, y1], mode='lines',
                        line=dict(color='#555566', width=1, dash='dot'),
                        hoverinfo='skip', showlegend=False,
                    ))

                fig.add_trace(go.Scatter(
                    x=corner_offset_x, y=corner_offset_y, mode='markers+text',
                    text=corner_labels, textposition='middle center',
                    textfont=dict(color='white', size=10, family='Arial Black'),
                    marker=dict(size=20, color=CORNER_MARKER_COLOR, line=dict(color='white', width=1)),
                    name='Corners', hoverinfo='skip', showlegend=False,
                ))

            fig.update_layout(
                plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
                title=dict(
                    text=(f"{selected_year} {selected_race} ({selected_session_type}) - "
                          f"{', '.join(telemetry_map.keys())}"),
                    font=dict(color='white', size=18),
                ),
                # Horizontal legend below the plot rather than Plotly's default
                # right-hand-side placement, which sat directly on top of the speed
                # colorbar (color-by-speed mode) - now they occupy different regions
                # regardless of how many legend entries there are.
                legend=dict(font=dict(color='white'), bgcolor=BG_COLOR,
                             orientation='h', yanchor='top', y=-0.02, xanchor='center', x=0.5),
                margin=dict(l=10, r=10, t=60, b=80),
                height=800,
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- CSV EXPORT --- whatever telemetry is currently plotted above (all
            # selected drivers/laps, full resolution - not the decimated display copy).
            export_df = pd.concat(
                [df.assign(Driver=d) for d, df in telemetry_map.items()],
                ignore_index=True,
            ) if telemetry_map else pd.DataFrame()
            if not export_df.empty:
                file_stub = f"{selected_year}_{selected_race}_{selected_session_type}".replace(' ', '_')
                st.download_button(
                    "Download telemetry as CSV",
                    data=export_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{file_stub}_telemetry.csv",
                    mime='text/csv',
                )
        except Exception as e:
            st.warning(f"Error plotting the track map: {e}")

        # --- THROTTLE & BRAKE --- own try/except: a failure here shouldn't take down
        # the track map above or the tyre panel below.
        # Same driver/lap selection as the track map above, plotted against Distance.
        # Colored by driver (matching the track map) rather than a fixed throttle/
        # brake color scheme, since with multiple drivers selected the important
        # distinction is who's who, not which channel is which - that's already
        # clear from the subplot titles.
        try:
            st.markdown("### Throttle & Brake")
            tb_fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                subplot_titles=('Throttle (%)', 'Brake'),
            )

            if is_single_lap_scrub:
                # Same client-side scrub technique as the track map above: one
                # persistent throttle/brake trace pair per driver, swapped per lap
                # via a Plotly frame - without this, Single Lap mode now loading
                # every quick lap up front (for the track map's scrubber) would
                # otherwise overlay every lap's throttle/brake trace at full
                # opacity here instead of showing just the current lap.
                all_lap_numbers = sorted(set(n for nums in plot_lap_numbers_map.values() for n in nums))
                throttle_idx, brake_idx = {}, {}
                for d, lap_numbers in plot_lap_numbers_map.items():
                    first_lap = lap_numbers[0]
                    first_tel = _decimate(telemetry_map[d][telemetry_map[d]['LapNumber'] == first_lap])
                    throttle_idx[d] = len(tb_fig.data)
                    tb_fig.add_trace(go.Scattergl(
                        x=first_tel['Distance'], y=first_tel['Throttle'], mode='lines',
                        line=dict(color=driver_colors[d], width=line_width, dash=driver_dash[d]),
                        name=d, legendgroup=f'driver_{d}',
                    ), row=1, col=1)
                    brake_idx[d] = len(tb_fig.data)
                    tb_fig.add_trace(go.Scattergl(
                        x=first_tel['Distance'], y=first_tel['Brake'].astype(float), mode='lines',
                        line=dict(color=driver_colors[d], width=line_width, dash=driver_dash[d]),
                        showlegend=False, legendgroup=f'driver_{d}',
                    ), row=2, col=1)

                frames = []
                for lap_number in all_lap_numbers:
                    frame_data, frame_traces = [], []
                    for d, lap_numbers in plot_lap_numbers_map.items():
                        frame_traces.append(throttle_idx[d])
                        frame_traces.append(brake_idx[d])
                        if lap_number in lap_numbers:
                            lap_tel = _decimate(telemetry_map[d][telemetry_map[d]['LapNumber'] == lap_number])
                            frame_data.append(go.Scattergl(x=lap_tel['Distance'], y=lap_tel['Throttle']))
                            frame_data.append(go.Scattergl(x=lap_tel['Distance'], y=lap_tel['Brake'].astype(float)))
                        else:
                            frame_data.append(go.Scattergl(x=[], y=[]))
                            frame_data.append(go.Scattergl(x=[], y=[]))
                    frames.append(go.Frame(name=str(lap_number), data=frame_data, traces=frame_traces))
                tb_fig.frames = frames

                tb_fig.update_layout(sliders=[dict(
                    active=0,
                    currentvalue=dict(prefix='Lap: ', font=dict(color='white')),
                    pad=dict(t=20), font=dict(color='white'),
                    steps=[dict(
                        method='animate',
                        args=[[str(lap_number)], dict(mode='immediate',
                                                       frame=dict(duration=0, redraw=True),
                                                       transition=dict(duration=0))],
                        label=str(lap_number),
                    ) for lap_number in all_lap_numbers],
                )])
            else:
                for d, telemetry_df in telemetry_map.items():
                    for i, (lap_number, lap_tel) in enumerate(telemetry_df.groupby('LapNumber')):
                        plot_tel = _decimate(lap_tel)
                        tb_fig.add_trace(go.Scattergl(
                            x=plot_tel['Distance'], y=plot_tel['Throttle'], mode='lines',
                            line=dict(color=driver_colors[d], width=line_width, dash=driver_dash[d]),
                            opacity=alpha_val,
                            name=d if i == 0 else None,
                            legendgroup=f'driver_{d}', showlegend=(i == 0),
                        ), row=1, col=1)
                        tb_fig.add_trace(go.Scattergl(
                            x=plot_tel['Distance'], y=plot_tel['Brake'].astype(float), mode='lines',
                            line=dict(color=driver_colors[d], width=line_width, dash=driver_dash[d]),
                            opacity=alpha_val, showlegend=False,
                            legendgroup=f'driver_{d}',
                        ), row=2, col=1)

            _style_subplot_figure(tb_fig, height=450)
            _apply_distance_or_corner_xaxis(tb_fig, corner_tick_vals, corner_tick_text, row=2, col=1)
            tb_fig.update_yaxes(title_text='Throttle %', color='white', gridcolor=TRACK_COLOR, row=1, col=1)
            tb_fig.update_yaxes(title_text='Brake', color='white', gridcolor=TRACK_COLOR, row=2, col=1)
            st.plotly_chart(tb_fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Error plotting throttle/brake: {e}")

        # --- TYRE DEGRADATION VS THROTTLE/BRAKE --- own try/except so a failure here
        # can't take down the charts above.
        # Independent of the lap selection above: pick a whole stint so brake/throttle
        # points can be compared across the tyre's full life, colored from fresh
        # (light) to worn (dark red) by TyreLife. Scoped to one driver at a time -
        # different drivers have different stints/compounds/lap numbers, so "tyre
        # degradation" doesn't have a single shared meaning across several drivers
        # the way the racing line and throttle/brake comparisons above do.
        try:
            st.markdown("### Tyre Degradation vs Throttle/Brake")
            tyre_candidates = list(telemetry_map.keys())
            tyre_driver = (tyre_candidates[0] if len(tyre_candidates) == 1
                           else st.selectbox("Driver for tyre analysis", tyre_candidates))
            stints = get_driver_stints(
                selected_year, selected_race, session_dict[selected_session_type], tyre_driver)

            if stints.empty:
                st.caption("No stint data available for this driver.")
            else:
                stint_labels = [
                    f"Stint {int(row['Stint'])} - {row['Compound']} "
                    f"(Laps {int(row['LapStart'])}-{int(row['LapEnd'])})"
                    for _, row in stints.iterrows()
                ]
                selected_stint_label = st.selectbox("Select Stint", stint_labels)
                selected_stint_num = int(stints.iloc[stint_labels.index(selected_stint_label)]['Stint'])

                stint_tel = get_stint_telemetry(
                    selected_year, selected_race, session_dict[selected_session_type],
                    tyre_driver, selected_stint_num)

                if stint_tel.empty:
                    st.caption("No telemetry available for this stint.")
                else:
                    min_life = stint_tel['TyreLife'].min()
                    life_span = max(stint_tel['TyreLife'].max() - min_life, 1)

                    tyre_fig = make_subplots(
                        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Throttle (%) by tyre age', 'Brake by tyre age'),
                    )
                    stint_laps_sorted = sorted(int(n) for n in stint_tel['LapNumber'].unique())
                    # Fade the full-stint overlay once there's more than a couple of laps,
                    # so the scrub highlight below (added at full opacity/width) actually
                    # stands out instead of blending into the color gradient.
                    context_alpha = 1.0 if len(stint_laps_sorted) <= 2 else 0.35
                    lap_colors = {}
                    context_trace_indices = []
                    for lap_number, lap_tel in stint_tel.groupby('LapNumber'):
                        life = lap_tel['TyreLife'].iloc[0]
                        frac = (life - min_life) / life_span
                        color = pcolors.sample_colorscale('YlOrRd', frac)[0]
                        lap_colors[int(lap_number)] = color
                        context_trace_indices.append(len(tyre_fig.data))
                        tyre_fig.add_trace(go.Scattergl(
                            x=lap_tel['Distance'], y=lap_tel['Throttle'], mode='lines',
                            line=dict(color=color, width=1.5), opacity=context_alpha,
                            name=f'Lap {int(lap_number)} (Tyre life {int(life)})',
                            legendgroup=f'lap{lap_number}',
                        ), row=1, col=1)
                        context_trace_indices.append(len(tyre_fig.data))
                        tyre_fig.add_trace(go.Scattergl(
                            x=lap_tel['Distance'], y=lap_tel['Brake'].astype(float), mode='lines',
                            line=dict(color=color, width=1.5), opacity=context_alpha, showlegend=False,
                            legendgroup=f'lap{lap_number}',
                        ), row=2, col=1)

                    # --- LAP SCRUB HIGHLIGHT --- a Plotly frame/slider (client-side, no
                    # Streamlit rerun) that isolates one lap at a time on top of the
                    # faded stint overlay above, so a single lap's throttle/brake shape
                    # is easy to pick out instead of reading it off an overlaid gradient.
                    if len(stint_laps_sorted) > 1:
                        first_lap = stint_laps_sorted[0]
                        first_tel = stint_tel[stint_tel['LapNumber'] == first_lap]
                        throttle_idx = len(tyre_fig.data)
                        tyre_fig.add_trace(go.Scattergl(
                            x=first_tel['Distance'], y=first_tel['Throttle'], mode='lines',
                            line=dict(color=lap_colors[first_lap], width=3.5),
                            name='Highlighted lap', legendgroup='highlight',
                        ), row=1, col=1)
                        brake_idx = len(tyre_fig.data)
                        tyre_fig.add_trace(go.Scattergl(
                            x=first_tel['Distance'], y=first_tel['Brake'].astype(float), mode='lines',
                            line=dict(color=lap_colors[first_lap], width=3.5), showlegend=False,
                            legendgroup='highlight',
                        ), row=2, col=1)

                        frames = []
                        for lap_number in stint_laps_sorted:
                            lap_tel = stint_tel[stint_tel['LapNumber'] == lap_number]
                            color = lap_colors[lap_number]
                            frames.append(go.Frame(
                                name=str(lap_number),
                                data=[
                                    go.Scattergl(x=lap_tel['Distance'], y=lap_tel['Throttle'],
                                                 line=dict(color=color)),
                                    go.Scattergl(x=lap_tel['Distance'], y=lap_tel['Brake'].astype(float),
                                                 line=dict(color=color)),
                                ],
                                traces=[throttle_idx, brake_idx],
                            ))
                        tyre_fig.frames = frames
                        tyre_fig.update_layout(
                            sliders=[dict(
                                active=0,
                                currentvalue=dict(prefix='Highlight lap: ', font=dict(color='white')),
                                pad=dict(t=20),
                                font=dict(color='white'),
                                steps=[dict(
                                    method='animate',
                                    args=[[str(lap_number)], dict(mode='immediate',
                                                                   frame=dict(duration=0, redraw=True),
                                                                   transition=dict(duration=0))],
                                    label=str(lap_number),
                                ) for lap_number in stint_laps_sorted],
                            )],
                            # Client-side toggle (Plotly restyle, no Streamlit rerun) between
                            # the full faded overlay and hiding every lap except whichever one
                            # the slider above is currently pointed at.
                            updatemenus=[dict(
                                type='buttons', direction='left',
                                x=0.0, xanchor='left', y=1.15, yanchor='top',
                                bgcolor=BG_COLOR, font=dict(color='white'),
                                buttons=[
                                    dict(label='Show all laps', method='restyle',
                                         args=[{'opacity': context_alpha}, context_trace_indices]),
                                    dict(label='Show only selected lap', method='restyle',
                                         args=[{'opacity': 0}, context_trace_indices]),
                                ],
                            )],
                        )

                    _style_subplot_figure(tyre_fig, height=500)
                    _apply_distance_or_corner_xaxis(tyre_fig, corner_tick_vals, corner_tick_text, row=2, col=1)
                    tyre_fig.update_yaxes(title_text='Throttle %', color='white', gridcolor=TRACK_COLOR,
                                           row=1, col=1)
                    tyre_fig.update_yaxes(title_text='Brake', color='white', gridcolor=TRACK_COLOR,
                                           row=2, col=1)
                    st.plotly_chart(tyre_fig, use_container_width=True)
                    st.caption(
                        "Line color shifts from light to dark red as the tyre ages within the stint "
                        "(darker = more worn). Use the slider under the plot to pick a lap, and the "
                        "'Show only selected lap' button above the plot to hide the rest of the stint "
                        "entirely - both work in the browser, no page refresh needed."
                    )
        except Exception as e:
            st.warning(f"Error plotting tyre degradation: {e}")


main_col, side_col = st.columns([3, 2])

with main_col:
    render_driver_dashboard(session)


def render_position_changes(session, year, race, session_type_label, session_type_code):
    # Race/Sprint only: Qualifying doesn't have a persistent running position across
    # a session the way a race does (it's knockout stages, not a continuous order).
    if session_type_code not in ('R', 'S'):
        return

    st.markdown("---")
    st.markdown("### Position Changes")

    with st.spinner("Building position-changes chart..."):
        try:
            laps, results = get_position_progression(year, race, session_type_code)
        except Exception as e:
            st.warning(f"Error loading position data: {e}")
            return

    if laps.empty or results.empty:
        st.caption("No lap-by-lap position data available for this session.")
        return

    results = results.dropna(subset=['GridPosition'])
    if results.empty:
        st.caption("No starting grid data available for this session.")
        return

    no_grid_data = sorted(set(laps['Driver'].unique()) - set(results['Driver']))
    if no_grid_data:
        st.caption(f"No starting grid data for: {', '.join(no_grid_data)} - excluded from this chart.")

    try:
        num_drivers = len(results)
        drop_position = num_drivers + 1
        max_lap = int(laps['LapNumber'].max())

        fig = go.Figure()
        for _, row in results.sort_values('GridPosition').iterrows():
            d = row['Driver']
            grid_pos = row['GridPosition']
            driver_laps = laps[laps['Driver'] == d].sort_values('LapNumber')

            # Start every line at its actual starting GRID slot (lap 0) - this is what
            # bakes penalties/pit-lane starts into the chart, rather than just
            # replaying qualifying order - then follow the real per-lap running
            # position from lap 1 onward.
            x_vals = [0] + driver_laps['LapNumber'].tolist()
            y_vals = [grid_pos] + driver_laps['Position'].tolist()

            if row['Status'] not in FINISHED_STATUSES and not driver_laps.empty:
                last_lap = int(driver_laps['LapNumber'].max())
                x_vals.append(last_lap + 1)
                y_vals.append(drop_position)

            color = get_driver_color_cached(year, race, session_type_code, d)
            fig.add_trace(go.Scattergl(
                x=x_vals, y=y_vals, mode='lines',
                line=dict(color=color, width=2),
                name=d,
                hovertemplate=f'{d}<br>Lap %{{x}}<br>P%{{y}}<extra></extra>',
            ))
            fig.add_annotation(
                x=0, y=grid_pos, text=d, showarrow=False,
                xanchor='right', xshift=-8,
                font=dict(color=color, size=11),
            )

        fig.update_layout(
            plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
            xaxis=dict(title='Lap', color='white', gridcolor=TRACK_COLOR, range=[-3, max_lap + 1]),
            yaxis=dict(title='Position', color='white', gridcolor=TRACK_COLOR,
                       dtick=1, range=[drop_position + 1, 0]),
            title=dict(
                text=f"{year} {race} ({session_type_label}) - Position Changes",
                font=dict(color='white', size=18),
            ),
            showlegend=False,
            margin=dict(l=60, r=10, t=60, b=10),
            height=700,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Starting position is the actual starting grid (after any penalties or pit-lane "
            "starts), not qualifying classification. A line dropping off the bottom marks a "
            "retirement (DNF/DNS) at that lap."
        )
    except Exception as e:
        st.warning(f"Error plotting position changes: {e}")


with main_col:
    render_position_changes(session, selected_year, selected_race, selected_session_type,
                             session_dict[selected_session_type])

with side_col:
    render_race_status_table(session, selected_year, selected_race, selected_session_type,
                              session_dict[selected_session_type])
