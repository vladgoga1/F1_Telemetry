
import os
from datetime import datetime

import fastf1
import fastf1.plotting
import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="F1 Telemetry", layout="centered")
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

    mask = np.zeros(len(tel), dtype=bool)
    mask[1:] = stuck | resync_tail
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


# --- 3. Cached Data Access Layer ---
# Session objects are large and mutable, so they're cached as a resource (kept live in
# memory by reference) rather than as data (which would pickle/copy them on every access).
@st.cache_resource(show_spinner=False)
def get_session_data(year, race, session_type):
    session = fastf1.get_session(year, race, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
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
def get_circuit_corners(year, race, session_type):
    session = get_session_data(year, race, session_type)
    circuit_info = session.get_circuit_info()
    return circuit_info.corners if circuit_info else None


@st.cache_data(show_spinner=False, persist="disk")
def get_driver_lap_telemetry(year, race, session_type, driver, lap_numbers):
    session = get_session_data(year, race, session_type)
    driver_laps = session.laps.pick_driver(driver).pick_quicklaps()
    laps_to_plot = driver_laps[driver_laps['LapNumber'].isin(lap_numbers)]
    frames = []
    for _, lap in laps_to_plot.iterlaps():
        tel = _drop_stuck_position_samples(lap.get_telemetry())[
            ['X', 'Y', 'Speed', 'Distance', 'Throttle', 'Brake']].copy()
        tel['LapNumber'] = lap['LapNumber']
        frames.append(tel)
    if not frames:
        return pd.DataFrame(columns=['X', 'Y', 'Speed', 'Distance', 'Throttle', 'Brake', 'LapNumber'])
    return pd.concat(frames, ignore_index=True)


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
    """
    session = get_session_data(year, race, session_type)
    driver_laps = session.laps.pick_driver(driver).pick_quicklaps()
    stint_laps = driver_laps[driver_laps['Stint'] == stint]
    frames = []
    for _, lap in stint_laps.iterlaps():
        tel = _drop_stuck_position_samples(lap.get_telemetry())[['Distance', 'Throttle', 'Brake']].copy()
        tel['LapNumber'] = lap['LapNumber']
        tel['TyreLife'] = lap['TyreLife']
        frames.append(tel)
    if not frames:
        return pd.DataFrame(columns=['Distance', 'Throttle', 'Brake', 'LapNumber', 'TyreLife'])
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner="Building track-limit corridor from every driver's fastest lap...", persist="disk")
def get_track_limit_envelope(year, race, session_type, bin_size=TRACK_LIMIT_BIN_SIZE_M):
    """Approximate legal-racing corridor: the min/max lateral offset (from the overall
    fastest lap's line) covered by each driver's own fastest clean lap. This is NOT the
    physical white-line/curb geometry FastF1 doesn't expose that - it's the spread of
    racing lines actually used at racing pace, which is what we use as the "normal"
    corridor to flag a driver running wider than the field on a given lap.
    """
    session = get_session_data(year, race, session_type)
    clean_laps = session.laps.pick_wo_box().pick_quicklaps()
    fastest_per_driver = clean_laps.loc[clean_laps.groupby('Driver')['LapTime'].idxmin()]

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

    min_by_driver = np.full((len(bins), n_drivers), np.inf)
    max_by_driver = np.full((len(bins), n_drivers), -np.inf)

    for _, lap in fastest_per_driver.iterlaps():
        tel = _drop_stuck_position_samples(lap.get_telemetry())
        px, py, pdist = tel['X'].to_numpy(), tel['Y'].to_numpy(), tel['Distance'].to_numpy()
        idx = np.clip(np.round(pdist / bin_size).astype(int), 0, len(bins) - 1)
        offset = (px - cx[idx]) * nx[idx] + (py - cy[idx]) * ny[idx]
        col = driver_col[lap['Driver']]
        np.minimum.at(min_by_driver[:, col], idx, offset)
        np.maximum.at(max_by_driver[:, col], idx, offset)

    min_by_driver[np.isinf(min_by_driver)] = np.nan
    max_by_driver[np.isinf(max_by_driver)] = np.nan

    # Trim roughly the single widest driver on each side per bin (a percentile scaled
    # to the field size) instead of taking the absolute extreme, so the corridor
    # reflects the field's normal spread rather than one outlier lap.
    trim_pct = 100.0 / n_drivers if n_drivers else 5.0
    with np.errstate(invalid='ignore'):
        min_off = np.nanpercentile(min_by_driver, trim_pct, axis=1)
        max_off = np.nanpercentile(max_by_driver, 100.0 - trim_pct, axis=1)

    min_off = pd.Series(min_off).ffill().bfill().to_numpy()
    max_off = pd.Series(max_off).ffill().bfill().to_numpy()

    return {
        'bins': bins, 'cx': cx, 'cy': cy, 'nx': nx, 'ny': ny,
        'min_off': min_off, 'max_off': max_off, 'bin_size': bin_size,
        'driver_count': len(fastest_per_driver),
    }


@st.cache_data(show_spinner=False, persist="disk")
def get_driver_track_limit_violations(year, race, session_type, driver, lap_numbers):
    telemetry_df = get_driver_lap_telemetry(year, race, session_type, driver, lap_numbers)
    if telemetry_df.empty:
        return telemetry_df.assign(Excess=[])

    envelope = get_track_limit_envelope(year, race, session_type)
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


with st.spinner("Downloading and processing session telemetry..."):
    session = get_session_data(selected_year, selected_race, session_dict[selected_session_type])


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
        view_mode = st.selectbox("View Mode", ["Single Lap", "Lap Interval", "All Laps Mixed (Average Trend)"])

    if not selected_drivers:
        st.warning("Select at least one driver.")
        st.stop()

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
        selected_lap = st.slider("Select Lap", min_value=min_lap, max_value=max_lap, value=min_lap)
        laps_to_plot_map = {d: laps[laps['LapNumber'] == selected_lap] for d, laps in driver_laps_map.items()}
        alpha_val = 1.0
        line_width = 2

    elif view_mode == "Lap Interval":
        selected_interval = st.slider("Select Lap Range", min_value=min_lap, max_value=max_lap,
                                       value=(min_lap, min(min_lap + 5, max_lap)))
        laps_to_plot_map = {
            d: laps[(laps['LapNumber'] >= selected_interval[0]) & (laps['LapNumber'] <= selected_interval[1])]
            for d, laps in driver_laps_map.items()
        }
        total_shown = sum(len(v) for v in laps_to_plot_map.values())
        alpha_val = max(0.3, 1.5 / total_shown) if total_shown > 0 else 1.0
        line_width = 1.5
        st.caption("Use the lap slider under the plot to scrub through individual laps in this "
                   "range - it plays back in the browser, no page refresh needed.")

    else:  # All Laps Mixed
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
            envelope = get_track_limit_envelope(
                selected_year, selected_race, session_dict[selected_session_type])
            edge_x = envelope['cx'] + envelope['nx'] * envelope['max_off']
            edge_y = envelope['cy'] + envelope['ny'] * envelope['max_off']
            other_edge_x = envelope['cx'] + envelope['nx'] * envelope['min_off']
            other_edge_y = envelope['cy'] + envelope['ny'] * envelope['min_off']

            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y, mode='lines',
                line=dict(color=CORRIDOR_COLOR, width=1), hoverinfo='skip', showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=other_edge_x, y=other_edge_y, mode='lines',
                line=dict(color=CORRIDOR_COLOR, width=1), fill='tonexty',
                fillcolor='rgba(58, 58, 85, 0.35)', hoverinfo='skip',
                name=f"Track-limit corridor ({envelope['driver_count']} drivers' fastest laps)",
            ))

            # --- PLOT EVERY SELECTED LAP (per driver) ---
            plot_lap_numbers_map = {
                d: tuple(sorted(laps['LapNumber'].astype(int).tolist()))
                for d, laps in laps_to_plot_map.items() if not laps.empty
            }
            telemetry_map = {
                d: get_driver_lap_telemetry(
                    selected_year, selected_race, session_dict[selected_session_type], d, lap_numbers)
                for d, lap_numbers in plot_lap_numbers_map.items()
            }

            for d, telemetry_df in telemetry_map.items():
                for i, (lap_number, lap_tel) in enumerate(telemetry_df.groupby('LapNumber')):
                    plot_tel = _decimate(lap_tel)
                    fig.add_trace(go.Scattergl(
                        x=plot_tel['X'], y=plot_tel['Y'], mode='lines',
                        line=dict(color=driver_colors[d], width=line_width, dash=driver_dash[d]),
                        opacity=alpha_val,
                        name=f'{d} Trajectory' if i == 0 else None,
                        legendgroup=f'driver_{d}', showlegend=(i == 0),
                        customdata=np.stack([plot_tel['Speed'], plot_tel['Distance'],
                                              [lap_number] * len(plot_tel)], axis=-1),
                        hovertemplate=(
                            f'{d} Lap %{{customdata[2]}}<br>'
                            'Speed: %{customdata[0]:.0f} km/h<br>'
                            'Distance: %{customdata[1]:.0f} m<extra></extra>'
                        ),
                    ))

            # --- LAP SCRUBBER (Lap Interval mode only) ---
            # A Plotly frame/slider, not a Streamlit widget: switching frames happens
            # entirely client-side in the browser, so scrubbing through laps doesn't
            # trigger a Streamlit rerun or page refresh. One scrub trace per driver;
            # frames are keyed by the UNION of lap numbers across drivers, since
            # different drivers can have different valid quick laps - a driver with no
            # data for a given frame's lap just goes empty for that frame.
            all_lap_numbers = sorted(set(n for nums in plot_lap_numbers_map.values() for n in nums))
            if view_mode == "Lap Interval" and len(all_lap_numbers) > 1:
                scrub_trace_idx = {}
                for d, lap_numbers in plot_lap_numbers_map.items():
                    scrub_trace_idx[d] = len(fig.data)
                    first_lap = lap_numbers[0]
                    first_tel = _decimate(telemetry_map[d][telemetry_map[d]['LapNumber'] == first_lap])
                    fig.add_trace(go.Scattergl(
                        x=first_tel['X'], y=first_tel['Y'], mode='lines',
                        line=dict(color=driver_colors[d], width=max(line_width * 2, 3), dash=driver_dash[d]),
                        name=f'{d} (scrub)',
                        customdata=np.stack([first_tel['Speed'], first_tel['Distance'],
                                              [first_lap] * len(first_tel)], axis=-1),
                        hovertemplate=(
                            f'{d} Lap %{{customdata[2]}}<br>'
                            'Speed: %{customdata[0]:.0f} km/h<br>'
                            'Distance: %{customdata[1]:.0f} m<extra></extra>'
                        ),
                    ))

                frames = []
                for lap_number in all_lap_numbers:
                    frame_data, frame_traces = [], []
                    for d, lap_numbers in plot_lap_numbers_map.items():
                        frame_traces.append(scrub_trace_idx[d])
                        if lap_number in lap_numbers:
                            lap_tel = _decimate(telemetry_map[d][telemetry_map[d]['LapNumber'] == lap_number])
                            frame_data.append(go.Scattergl(
                                x=lap_tel['X'], y=lap_tel['Y'],
                                customdata=np.stack([lap_tel['Speed'], lap_tel['Distance'],
                                                      [lap_number] * len(lap_tel)], axis=-1),
                            ))
                        else:
                            frame_data.append(go.Scattergl(x=[], y=[], customdata=np.empty((0, 3))))
                    frames.append(go.Frame(name=str(lap_number), data=frame_data, traces=frame_traces))
                fig.frames = frames

                fig.update_layout(
                    sliders=[dict(
                        active=0,
                        currentvalue=dict(prefix='Scrub lap: ', font=dict(color='white')),
                        pad=dict(t=20),
                        font=dict(color='white'),
                        steps=[dict(
                            method='animate',
                            args=[[str(lap_number)], dict(mode='immediate',
                                                           frame=dict(duration=0, redraw=True),
                                                           transition=dict(duration=0))],
                            label=str(lap_number),
                        ) for lap_number in all_lap_numbers],
                    )],
                )

            # --- TRACK LIMIT VIOLATIONS (per driver) ---
            # Points where a selected driver ran wider than every other driver's own
            # fastest lap at that point on track.
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
                st.caption("All selected drivers stayed within the fastest-lap corridor for the selected lap(s).")

            # --- SECTOR GATES ---
            add_sector_gate(fig, ref_tel, sf_time, SECTOR_COLORS['Start/Finish'], 'Start/Finish')
            add_sector_gate(fig, ref_tel, sector1_time, SECTOR_COLORS['Sector 1 Split'], 'Sector 1 Split')
            add_sector_gate(fig, ref_tel, sector2_time, SECTOR_COLORS['Sector 2 Split'], 'Sector 2 Split')

            # --- TURN NUMBERS ---
            corners = get_circuit_corners(
                selected_year, selected_race, session_dict[selected_session_type])
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
                offset_x = corners['X'] + offset_dist * np.cos(angle_rad)
                offset_y = corners['Y'] + offset_dist * np.sin(angle_rad)
                labels = [
                    f"{int(row['Number'])}{row['Letter'] if pd.notna(row['Letter']) else ''}"
                    for _, row in corners.iterrows()
                ]

                for x0, y0, x1, y1 in zip(corners['X'], corners['Y'], offset_x, offset_y):
                    fig.add_trace(go.Scattergl(
                        x=[x0, x1], y=[y0, y1], mode='lines',
                        line=dict(color='#555566', width=1, dash='dot'),
                        hoverinfo='skip', showlegend=False,
                    ))

                fig.add_trace(go.Scatter(
                    x=offset_x, y=offset_y, mode='markers+text',
                    text=labels, textposition='middle center',
                    textfont=dict(color='white', size=10, family='Arial Black'),
                    marker=dict(size=20, color=CORNER_MARKER_COLOR, line=dict(color='white', width=1)),
                    name='Corners', hoverinfo='skip', showlegend=False,
                ))

                # Distance-along-lap nearest each corner (via the reference lap's own
                # Distance channel), so the throttle/brake plot below can label its x-axis
                # by corner instead of raw metres.
                ref_x, ref_y, ref_d = ref_tel['X'].to_numpy(), ref_tel['Y'].to_numpy(), ref_tel['Distance'].to_numpy()
                corner_dists = [
                    ref_d[np.argmin((ref_x - row['X']) ** 2 + (ref_y - row['Y']) ** 2)]
                    for _, row in corners.iterrows()
                ]
                order = np.argsort(corner_dists)
                corner_tick_vals = [corner_dists[i] for i in order]
                corner_tick_text = [labels[i] for i in order]

            fig.update_layout(
                plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
                title=dict(
                    text=(f"{selected_year} {selected_race} ({selected_session_type}) - "
                          f"{', '.join(telemetry_map.keys())}"),
                    font=dict(color='white', size=18),
                ),
                legend=dict(font=dict(color='white'), bgcolor=BG_COLOR),
                margin=dict(l=10, r=10, t=60, b=10),
                height=800,
            )

            st.plotly_chart(fig, use_container_width=True)

            # --- THROTTLE & BRAKE ---
            # Same driver/lap selection as the track map above, plotted against Distance.
            # Colored by driver (matching the track map) rather than a fixed throttle/
            # brake color scheme, since with multiple drivers selected the important
            # distinction is who's who, not which channel is which - that's already
            # clear from the subplot titles.
            st.markdown("### Throttle & Brake")
            tb_fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                subplot_titles=('Throttle (%)', 'Brake'),
            )
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

            tb_fig.update_layout(
                plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
                legend=dict(font=dict(color='white'), bgcolor=BG_COLOR),
                margin=dict(l=10, r=10, t=40, b=10), height=450,
                font=dict(color='white'),
            )
            if corner_tick_vals:
                tb_fig.update_xaxes(title_text='Corner', tickmode='array', tickvals=corner_tick_vals,
                                     ticktext=corner_tick_text, color='white', gridcolor=TRACK_COLOR, row=2, col=1)
            else:
                tb_fig.update_xaxes(title_text='Distance (m)', color='white', gridcolor=TRACK_COLOR, row=2, col=1)
            tb_fig.update_yaxes(title_text='Throttle %', color='white', gridcolor=TRACK_COLOR, row=1, col=1)
            tb_fig.update_yaxes(title_text='Brake', color='white', gridcolor=TRACK_COLOR, row=2, col=1)
            st.plotly_chart(tb_fig, use_container_width=True)

            # --- TYRE DEGRADATION VS THROTTLE/BRAKE ---
            # Independent of the lap selection above: pick a whole stint so brake/throttle
            # points can be compared across the tyre's full life, colored from fresh
            # (light) to worn (dark red) by TyreLife. Scoped to one driver at a time -
            # different drivers have different stints/compounds/lap numbers, so "tyre
            # degradation" doesn't have a single shared meaning across several drivers
            # the way the racing line and throttle/brake comparisons above do.
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
                    for lap_number, lap_tel in stint_tel.groupby('LapNumber'):
                        life = lap_tel['TyreLife'].iloc[0]
                        frac = (life - min_life) / life_span
                        color = pcolors.sample_colorscale('YlOrRd', frac)[0]
                        tyre_fig.add_trace(go.Scattergl(
                            x=lap_tel['Distance'], y=lap_tel['Throttle'], mode='lines',
                            line=dict(color=color, width=1.5),
                            name=f'Lap {int(lap_number)} (Tyre life {int(life)})',
                            legendgroup=f'lap{lap_number}',
                        ), row=1, col=1)
                        tyre_fig.add_trace(go.Scattergl(
                            x=lap_tel['Distance'], y=lap_tel['Brake'].astype(float), mode='lines',
                            line=dict(color=color, width=1.5), showlegend=False,
                            legendgroup=f'lap{lap_number}',
                        ), row=2, col=1)

                    tyre_fig.update_layout(
                        plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
                        legend=dict(font=dict(color='white'), bgcolor=BG_COLOR),
                        margin=dict(l=10, r=10, t=40, b=10), height=500,
                        font=dict(color='white'),
                    )
                    if corner_tick_vals:
                        tyre_fig.update_xaxes(title_text='Corner', tickmode='array', tickvals=corner_tick_vals,
                                               ticktext=corner_tick_text, color='white', gridcolor=TRACK_COLOR,
                                               row=2, col=1)
                    else:
                        tyre_fig.update_xaxes(title_text='Distance (m)', color='white', gridcolor=TRACK_COLOR,
                                               row=2, col=1)
                    tyre_fig.update_yaxes(title_text='Throttle %', color='white', gridcolor=TRACK_COLOR,
                                           row=1, col=1)
                    tyre_fig.update_yaxes(title_text='Brake', color='white', gridcolor=TRACK_COLOR,
                                           row=2, col=1)
                    st.plotly_chart(tyre_fig, use_container_width=True)
                    st.caption(
                        "Line color shifts from light to dark red as the tyre ages within the stint "
                        "(darker = more worn). Compare traces across laps to see how braking points "
                        "and throttle application shift as the tyre degrades."
                    )
        except Exception as e:
            st.warning(f"Error plotting data: {e}")


render_driver_dashboard(session)
