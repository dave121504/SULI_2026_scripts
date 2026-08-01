#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pyart
import s3fs
from cartopy.geodesic import Geodesic
from cartopy.mpl.ticker import LongitudeFormatter
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from metpy.plots import USCOUNTIES
from PIL import Image, ImageEnhance, ImageOps
from shapely.geometry import box, shape


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------

RADAR_SITE = "KLOT"

# Argonne Testbed for Multiscale Observational Science (ATMOS)
ATMOS_LAT = 41.70121
ATMOS_LON = -87.99495

# The GIF is placed in an easy-to-find Downloads subfolder. Raw radar files are
# not retained.
OUTPUT_DIR = Path.home() / "Downloads" / "radar_qpe_output"

# Number of recent complete radar volumes included in the animation.
NUMBER_OF_SCANS = 10

# GIF timing and resolution. The final frame pauses longer before looping.
FRAME_DURATION_MS = 650
FINAL_FRAME_DURATION_MS = 1500
GIF_DPI = 140

# Consistent, presentation-friendly typography for axes and colorbars.
MAP_COORDINATE_FONT_SIZE = 11
AXIS_LABEL_FONT_SIZE = 14
AXIS_TICK_FONT_SIZE = 12
COLORBAR_LABEL_FONT_SIZE = 13
LOWER_PANEL_TITLE_FONT_SIZE = 16

# Lowest-elevation PPI sweep.
SWEEP = 0

# Fixed settings for the gridded pseudo-RHI and ATMOS column panels.
RHI_MINIMUM_DBZ = -10.0
RHI_HALF_LENGTH_KM = 60.0
RHI_MAX_HEIGHT_KM = 12.0
RHI_GRID_HORIZONTAL_LIMIT_KM = 70.0
RHI_GRID_HORIZONTAL_POINTS = 141  # 1 km grid spacing
RHI_GRID_VERTICAL_POINTS = 49  # 250 m grid spacing
RHI_CROSS_SECTION_STEPS = 241  # 0.5 km along-section spacing
WIND_PROFILE_MAX_HEIGHT_KM = 5.0
WIND_PROFILE_HEIGHT_SPACING_M = 100.0
WIND_DISPLAY_MAX_HEIGHT_KFT = 16.0
COLUMN_AZIMUTH_SPREAD = 3
COLUMN_SPATIAL_SPREAD = 3
DEALIAS_VELOCITY_FOR_COLUMN = True
KNOT_TO_MPH = 1.150779448
M_S_TO_KNOTS = 1.943844
# Keep the wind-barb colors directly comparable across scans and GIF frames.
WIND_SPEED_MIN_MPH = 0.0
WIND_SPEED_MAX_MPH = 140.0
WIND_CALM_MAX_MPH = 5.0
WIND_SPEED_COLORMAP = "SpectralExtended"
WIND_SPECTRAL_COLOR_START_FRACTION = 0.20
WIND_SPEED_COLORBAR_TICKS_MPH = (
    0.0,
    20.0,
    40.0,
    60.0,
    80.0,
    100.0,
    120.0,
    140.0,
)

# Neighborhood used to estimate conditions at ATMOS.
SAMPLE_RADIUS_KM = 1.0

# Reflectivity and correlation-coefficient quality control.
MINIMUM_DBZ = 5.0
MAXIMUM_DBZ = 53.0
RHOHV_MINIMUM = 0.80

# Py-ART reflectivity-rain-rate relationship: R = ALPHA * Z**BETA.
ALPHA = 0.0376
BETA = 0.6112

# Search today plus this many preceding UTC dates. This handles UTC midnight,
# brief archive delays, and short radar outages.
LOOKBACK_DAYS = 3

# NOAA/Unidata's public NEXRAD Level-II archive on Amazon S3. The former
# noaa-nexrad-level2 bucket was retired in 2025.
NEXRAD_BUCKET = "unidata-nexrad-level2"

# NWS warning polygons are retrieved once per run and held in memory. The
# historical alerts endpoint contains the preceding seven days, which covers
# this script's radar lookback.
NWS_ALERTS_URL = "https://api.weather.gov/alerts"
NWS_ALERT_AREAS = "IL,IN"
NWS_ALERT_USER_AGENT = (
    "KLOT-ATMOS-radar-animation/1.0 (educational research)"
)
NWS_ALERT_TIMEOUT_SECONDS = 15
NWS_ALERT_HISTORY_HOURS = 72
NWS_ALERT_QUERY_EVENTS = (
    "Tornado Warning",
    "Severe Thunderstorm Warning",
    "Flash Flood Warning",
    "Flood Warning",
    "Lakeshore Flood Warning",
)
WARNING_STYLES = {
    "tornado": {
        "color": "red",
        "label": "Tornado Warning",
    },
    "severe": {
        "color": "yellow",
        "label": "Severe Thunderstorm Warning",
    },
    "flood": {
        "color": "#00C853",
        "label": "Flood Warning",
    },
}

# City markers shown on the regional panel. Label offsets are in degrees and
# keep nearby labels from sitting directly on their markers.
REGIONAL_CITIES = (
    ("Chicago", 41.8781, -87.6298, 0.018, 0.018, "left"),
    ("Joliet", 41.5250, -88.0817, 0.018, -0.024, "left"),
    ("Naperville", 41.7508, -88.1535, 0.018, 0.018, "left"),
    ("Aurora", 41.7606, -88.3201, -0.018, 0.018, "right"),
    ("Hammond", 41.5834, -87.5000, 0.018, -0.024, "left"),
    ("Waukegan", 42.3636, -87.8448, 0.018, 0.018, "left"),
    ("Schaumburg", 42.0334, -88.0834, 0.018, 0.018, "left"),
    ("Woodstock", 42.31377, -88.44830, -0.018, 0.018, "right")
)

# Smaller labels used on the detailed ATMOS panel.
ZOOM_CITIES = (
    ("Darien", 41.7517, -87.9737, -0.001, -0.004, "right"),
    ("Woodridge", 41.7470, -88.0503, 0.003, -0.004, "left"),
    ("Lemont", 41.6736, -88.0017, 0.003, 0.003, "left"),
)


class GrayscaleOSM(cimgt.OSM):
    """OpenStreetMap tiles converted to a light, muted grayscale."""

    def __init__(self):
        super().__init__(cache=False)
        self._memory_cache = {}

    def get_image(self, tile):
        cache_key = tuple(tile)
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        image, extent, origin = super().get_image(tile)
        image = ImageOps.grayscale(image).convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(0.72)
        image = ImageEnhance.Brightness(image).enhance(1.12)
        result = (image, extent, origin)
        self._memory_cache[cache_key] = result
        return result


def scan_time_from_name(object_name: str) -> datetime | None:
    """Extract a timezone-aware UTC scan time from a NEXRAD filename."""
    filename = Path(object_name).name
    match = re.match(
        rf"^{re.escape(RADAR_SITE)}(\d{{8}})_(\d{{6}})(?:_V\d+)?(?:\..*)?$",
        filename,
    )
    if match is None:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def make_wind_speed_colormap() -> LinearSegmentedColormap:
    """Keep calm winds black and color winds of 5 mph or greater."""
    base_colormap = plt.get_cmap(WIND_SPEED_COLORMAP)
    calm_fraction = (
        (WIND_CALM_MAX_MPH - WIND_SPEED_MIN_MPH)
        / (WIND_SPEED_MAX_MPH - WIND_SPEED_MIN_MPH)
    )
    color_points = [
        (0.0, "black"),
        (np.nextafter(calm_fraction, 0.0), "black"),
    ]
    for sample_fraction in np.linspace(0.0, 1.0, 256):
        color_points.append(
            (
                calm_fraction + (1.0 - calm_fraction) * sample_fraction,
                base_colormap(
                    WIND_SPECTRAL_COLOR_START_FRACTION
                    + (1.0 - WIND_SPECTRAL_COLOR_START_FRACTION)
                    * sample_fraction
                ),
            )
        )
    return LinearSegmentedColormap.from_list(
        "ATMOS_column_adjusted_SpectralExtended_mph",
        color_points,
        N=512,
    )


def find_latest_nexrad_objects(count: int) -> list[tuple[str, datetime]]:
    """Return public S3 keys and times for the newest KLOT volumes."""
    filesystem = s3fs.S3FileSystem(anon=True, use_listings_cache=False)
    now = datetime.now(timezone.utc)
    candidates: list[tuple[datetime, str, int | None]] = []

    for day_offset in range(LOOKBACK_DAYS):
        day = (now - timedelta(days=day_offset)).date()
        prefix = (
            f"{NEXRAD_BUCKET}/{day:%Y}/{day:%m}/{day:%d}/{RADAR_SITE}"
        )
        try:
            filesystem.invalidate_cache(prefix)
            entries = filesystem.ls(prefix, detail=True)
        except FileNotFoundError:
            continue
        except Exception as exc:
            raise RuntimeError(
                "Unable to list NOAA radar files. Check the internet connection "
                "and confirm that s3fs is installed in the active environment."
            ) from exc

        for entry in entries:
            if isinstance(entry, str):
                object_name = entry
                object_size = None
            else:
                object_name = str(entry["name"])
                size_value = entry.get("size")
                object_size = int(size_value) if size_value is not None else None

            scan_time = scan_time_from_name(object_name)
            if scan_time is not None and (object_size is None or object_size > 0):
                candidates.append((scan_time, object_name, object_size))

    if not candidates:
        raise FileNotFoundError(
            f"No complete {RADAR_SITE} volumes were found during the last "
            f"{LOOKBACK_DAYS} UTC dates."
        )

    ordered = sorted(candidates, key=lambda item: item[0])
    selected = ordered[-count:]
    if len(selected) < count:
        print(
            f"WARNING: requested {count} scans but found only {len(selected)} "
            f"during the last {LOOKBACK_DAYS} UTC dates.",
            file=sys.stderr,
        )
    return [(object_name, scan_time) for scan_time, object_name, _ in selected]


def parse_nws_datetime(value: object) -> datetime | None:
    """Parse an NWS ISO-8601 timestamp as a timezone-aware UTC datetime."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def warning_category(event: object) -> str | None:
    """Map NWS warning event names to the requested display categories."""
    if not isinstance(event, str):
        return None
    if event == "Tornado Warning":
        return "tornado"
    if event == "Severe Thunderstorm Warning":
        return "severe"
    if event.endswith("Flood Warning"):
        return "flood"
    return None


def warning_vtec_identity(
    properties: dict[str, object],
) -> tuple[str, str | None]:
    """Return a lifecycle identity and VTEC action for an alert message."""
    parameters = properties.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    vtec_values = parameters.get("VTEC", [])
    if isinstance(vtec_values, str):
        vtec_values = [vtec_values]
    if not isinstance(vtec_values, list):
        vtec_values = []

    vtec_pattern = re.compile(
        r"/O\.(?P<action>[A-Z]{3})\."
        r"(?P<office>[A-Z]{4})\."
        r"(?P<phenomena>[A-Z]{2})\."
        r"(?P<significance>[A-Z])\."
        r"(?P<event_number>\d{4})\."
    )
    for value in vtec_values:
        if not isinstance(value, str):
            continue
        match = vtec_pattern.search(value)
        if match is None:
            continue
        identity = ".".join(
            (
                match.group("office"),
                match.group("phenomena"),
                match.group("significance"),
                match.group("event_number"),
            )
        )
        return identity, match.group("action")

    fallback_id = str(properties.get("id") or properties.get("@id") or "")
    return fallback_id, None


def fetch_nws_warning_records(
    scan_times: list[datetime],
) -> list[dict[str, object]]:
    """Retrieve relevant NWS warning messages once for all GIF frames."""
    if not scan_times:
        return []

    query_start = min(scan_times) - timedelta(
        hours=NWS_ALERT_HISTORY_HOURS
    )
    query_end = max(scan_times) + timedelta(minutes=5)
    query = urllib.parse.urlencode(
        {
            "start": query_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": query_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "status": "actual",
            "message_type": "alert,update,cancel",
            "event": ",".join(NWS_ALERT_QUERY_EVENTS),
            "area": NWS_ALERT_AREAS,
            "limit": 500,
        },
        safe=",",
    )
    request = urllib.request.Request(
        f"{NWS_ALERTS_URL}?{query}",
        headers={
            "Accept": "application/geo+json",
            "User-Agent": NWS_ALERT_USER_AGENT,
        },
    )
    with urllib.request.urlopen(
        request,
        timeout=NWS_ALERT_TIMEOUT_SECONDS,
    ) as response:
        payload = json.load(response)

    records: list[dict[str, object]] = []
    for feature in payload.get("features", []):
        if not isinstance(feature, dict):
            continue
        properties = feature.get("properties")
        if not isinstance(properties, dict):
            continue
        category = warning_category(properties.get("event"))
        if category is None:
            continue

        lifecycle_id, vtec_action = warning_vtec_identity(properties)
        if not lifecycle_id:
            lifecycle_id = str(feature.get("id") or "")
        if not lifecycle_id:
            continue

        geometry_mapping = feature.get("geometry")
        geometry = None
        if isinstance(geometry_mapping, dict):
            try:
                geometry = shape(geometry_mapping)
            except Exception:
                geometry = None

        records.append(
            {
                "lifecycle_id": lifecycle_id,
                "vtec_action": vtec_action,
                "message_type": properties.get("messageType"),
                "category": category,
                "event": properties.get("event"),
                "sent": parse_nws_datetime(properties.get("sent")),
                "effective": parse_nws_datetime(
                    properties.get("effective")
                ),
                "onset": parse_nws_datetime(properties.get("onset")),
                "ends": parse_nws_datetime(properties.get("ends")),
                "expires": parse_nws_datetime(properties.get("expires")),
                "geometry": geometry,
            }
        )

    return records


def warnings_valid_at(
    warning_records: list[dict[str, object]],
    scan_time: datetime,
) -> list[dict[str, object]]:
    """Select the latest active message for each warning at a scan time."""
    latest_by_lifecycle: dict[str, dict[str, object]] = {}
    for record in warning_records:
        sent = record.get("sent")
        if not isinstance(sent, datetime) or sent > scan_time:
            continue
        lifecycle_id = str(record["lifecycle_id"])
        previous = latest_by_lifecycle.get(lifecycle_id)
        previous_sent = previous.get("sent") if previous else None
        if not isinstance(previous_sent, datetime) or sent >= previous_sent:
            latest_by_lifecycle[lifecycle_id] = record

    active_records: list[dict[str, object]] = []
    for record in latest_by_lifecycle.values():
        if record.get("message_type") == "Cancel":
            continue
        if record.get("vtec_action") in {"CAN", "EXP", "UPG"}:
            continue

        start_time = (
            record.get("onset")
            or record.get("effective")
            or record.get("sent")
        )
        end_time = record.get("ends") or record.get("expires")
        if isinstance(start_time, datetime) and scan_time < start_time:
            continue
        if isinstance(end_time, datetime) and scan_time > end_time:
            continue
        if record.get("geometry") is None:
            continue
        active_records.append(record)

    return active_records


def add_warning_overlays(
    axis,
    geographic_crs,
    warning_records: list[dict[str, object]],
    scan_time: datetime,
    extent: list[float],
) -> set[str]:
    """Draw scan-time-valid NWS warning outlines on one map axis."""
    map_bounds = box(extent[0], extent[2], extent[1], extent[3])
    categories: set[str] = set()

    for record in warnings_valid_at(warning_records, scan_time):
        geometry = record["geometry"]
        if not geometry.intersects(map_bounds):
            continue
        category = str(record["category"])
        style = WARNING_STYLES[category]
        categories.add(category)

        # A thin black underlay keeps yellow and green polygons visible over
        # both the grayscale basemap and similarly colored radar echoes.
        axis.add_geometries(
            [geometry],
            crs=geographic_crs,
            facecolor="none",
            edgecolor="black",
            linewidth=4.0,
            linestyle="-",
            zorder=11,
        )
        axis.add_geometries(
            [geometry],
            crs=geographic_crs,
            facecolor="none",
            edgecolor=style["color"],
            linewidth=2.4,
            linestyle="-",
            zorder=12,
        )

    return categories


def add_map_legend(axis, warning_categories: set[str]) -> None:
    """Add existing map entries and warning-line proxies to one legend."""
    handles, labels = axis.get_legend_handles_labels()
    for category in ("tornado", "severe", "flood"):
        if category not in warning_categories:
            continue
        style = WARNING_STYLES[category]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linewidth=3.0,
                linestyle="-",
                label=style["label"],
            )
        )
        labels.append(str(style["label"]))

    axis.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        framealpha=0.9,
        fontsize=8,
    )


def radar_datetime_utc(radar) -> datetime:
    """Return the first radar-ray time as a timezone-aware UTC datetime."""
    units = radar.time["units"]
    calendar = radar.time.get("calendar", "standard")
    value = float(np.asarray(radar.time["data"])[0])
    converted = netCDF4.num2date(
        value,
        units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=False,
    )
    converted = datetime(
        converted.year,
        converted.month,
        converted.day,
        converted.hour,
        converted.minute,
        converted.second,
        converted.microsecond,
    )
    return converted.replace(tzinfo=timezone.utc)


def find_field(radar, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate field present in a Radar object."""
    return next((name for name in candidates if name in radar.fields), None)


def reflectivity_field(radar) -> str:
    """Detect the reflectivity field name."""
    field = find_field(
        radar,
        (
            "reflectivity",
            "corrected_reflectivity",
            "reflectivity_horizontal",
            "DBZ",
            "DBZH",
            "equivalent_reflectivity_factor",
        ),
    )
    if field is None:
        available = ", ".join(sorted(radar.fields))
        raise KeyError(f"No reflectivity field was found. Available fields: {available}")
    return field


def correlation_field(radar) -> str | None:
    """Detect the copolar correlation-coefficient field, if available."""
    return find_field(
        radar,
        ("cross_correlation_ratio", "copol_correlation_coeff", "RHOHV", "rhohv"),
    )


def velocity_field(radar) -> str | None:
    """Detect the Doppler radial-velocity field, if available."""
    return find_field(
        radar,
        (
            "corrected_velocity",
            "velocity",
            "radial_velocity",
            "mean_doppler_velocity",
            "VEL",
            "VRADH",
        ),
    )


def haversine_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    site_latitude: float,
    site_longitude: float,
) -> np.ndarray:
    """Calculate great-circle distance from radar gates to one site."""
    earth_radius_km = 6371.0
    latitude_radians = np.deg2rad(latitudes)
    site_latitude_radians = np.deg2rad(site_latitude)
    latitude_difference = site_latitude_radians - latitude_radians
    longitude_difference = np.deg2rad(site_longitude) - np.deg2rad(longitudes)
    haversine = (
        np.sin(latitude_difference / 2.0) ** 2
        + np.cos(latitude_radians)
        * np.cos(site_latitude_radians)
        * np.sin(longitude_difference / 2.0) ** 2
    )
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(haversine))


def estimate_atmos_rain_rate(radar, radar_source: str) -> dict[str, object]:
    """Estimate instantaneous rain rate in a neighborhood around ATMOS."""
    if SWEEP < 0 or SWEEP >= radar.nsweeps:
        raise IndexError(f"Sweep {SWEEP} is invalid; radar has {radar.nsweeps} sweeps.")

    reflectivity_name = reflectivity_field(radar)
    sweep_slice = radar.get_slice(SWEEP)
    reflectivity = np.ma.asarray(
        radar.get_field(SWEEP, reflectivity_name, copy=False)
    )
    gate_latitude, gate_longitude, gate_altitude = radar.get_gate_lat_lon_alt(SWEEP)
    distance = haversine_km(
        gate_latitude,
        gate_longitude,
        ATMOS_LAT,
        ATMOS_LON,
    )
    neighborhood = distance <= SAMPLE_RADIUS_KM

    rain_field = pyart.retrieve.est_rain_rate_z(
        radar,
        alpha=ALPHA,
        beta=BETA,
        refl_field=reflectivity_name,
    )
    rain_rate = np.ma.asarray(rain_field["data"][sweep_slice])

    reflectivity_values = np.ma.filled(reflectivity, np.nan).astype(float)
    rain_values = np.ma.filled(rain_rate, np.nan).astype(float)
    usable = (
        neighborhood
        & np.isfinite(reflectivity_values)
        & np.isfinite(rain_values)
    )

    rhohv_name = correlation_field(radar)
    if rhohv_name is not None:
        rhohv = np.ma.asarray(radar.get_field(SWEEP, rhohv_name, copy=False))
        rhohv_values = np.ma.filled(rhohv, np.nan).astype(float)
        if np.any(neighborhood & np.isfinite(rhohv_values)):
            usable &= np.isfinite(rhohv_values) & (rhohv_values >= RHOHV_MINIMUM)

    rainy_gates = usable & (reflectivity_values >= MINIMUM_DBZ)
    if np.any(rainy_gates):
        maximum_rate = ALPHA * (10.0 ** (MAXIMUM_DBZ / 10.0)) ** BETA
        site_rate = float(
            np.nanmedian(np.minimum(rain_values[rainy_gates], maximum_rate))
        )
        site_dbz = float(
            np.nanmedian(np.minimum(reflectivity_values[rainy_gates], MAXIMUM_DBZ))
        )
        site_altitude_m = float(np.nanmedian(gate_altitude[rainy_gates]))
        gates_used = int(np.count_nonzero(rainy_gates))
    else:
        site_rate = 0.0
        site_dbz = np.nan
        site_altitude_m = np.nan
        gates_used = 0

    return {
        "scan_time_utc": radar_datetime_utc(radar),
        "radar_source": radar_source,
        "reflectivity_field": reflectivity_name,
        "rhohv_field": rhohv_name or "not available",
        "median_reflectivity_dbz": site_dbz,
        "rain_rate_mm_h": site_rate,
        "median_gate_altitude_m": site_altitude_m,
        "gates_used": gates_used,
        "sample_radius_km": SAMPLE_RADIUS_KM,
    }


def make_gatefilter(radar, reflectivity_name: str):
    """Construct a display-only reflectivity quality-control filter."""
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_invalid(reflectivity_name)
    gatefilter.exclude_below(reflectivity_name, MINIMUM_DBZ)

    rhohv_name = correlation_field(radar)
    if rhohv_name is not None:
        rhohv = np.ma.asarray(radar.get_field(SWEEP, rhohv_name, copy=False))
        if np.ma.count(rhohv) > 0:
            gatefilter.exclude_below(rhohv_name, RHOHV_MINIMUM)
    return gatefilter


def make_rhi_gatefilter(radar, reflectivity_name: str):
    """Filter invalid/very weak RHI gates without precipitation-only RHOHV QC."""
    gatefilter = pyart.filters.GateFilter(radar)
    gatefilter.exclude_invalid(reflectivity_name)
    gatefilter.exclude_below(reflectivity_name, RHI_MINIMUM_DBZ)
    return gatefilter


def add_map_features(axis) -> None:
    """Add geographic reference features to a Cartopy map axis."""
    axis.add_feature(
        cfeature.LAKES.with_scale("10m"),
        facecolor="none",
        edgecolor="0.35",
        linewidth=0.6,
    )
    axis.add_feature(
        cfeature.STATES.with_scale("10m"), edgecolor="0.25", linewidth=0.9
    )
    axis.coastlines(resolution="10m", color="0.3", linewidth=0.6)


def add_city_labels(
    axis,
    geographic_crs,
    cities,
    marker_size: float,
    font_size: float,
) -> None:
    """Add selected city markers and high-contrast labels to a map."""
    for name, latitude, longitude, dx, dy, horizontal_alignment in cities:
        axis.scatter(
            longitude,
            latitude,
            marker="o",
            s=marker_size,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            transform=geographic_crs,
            zorder=13,
        )
        axis.text(
            longitude + dx,
            latitude + dy,
            name,
            transform=geographic_crs,
            ha=horizontal_alignment,
            va="center",
            fontsize=font_size,
            fontweight="bold",
            color="black",
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
            zorder=14,
        )


def bearing_degrees(
    start_latitude: float,
    start_longitude: float,
    end_latitude: float,
    end_longitude: float,
) -> float:
    """Return the initial great-circle bearing from one point to another."""
    start_latitude_rad = np.deg2rad(start_latitude)
    end_latitude_rad = np.deg2rad(end_latitude)
    longitude_difference = np.deg2rad(end_longitude - start_longitude)
    numerator = np.sin(longitude_difference) * np.cos(end_latitude_rad)
    denominator = (
        np.cos(start_latitude_rad) * np.sin(end_latitude_rad)
        - np.sin(start_latitude_rad)
        * np.cos(end_latitude_rad)
        * np.cos(longitude_difference)
    )
    return float((np.rad2deg(np.arctan2(numerator, denominator)) + 360.0) % 360.0)


def destination_lat_lon(
    start_latitude: float,
    start_longitude: float,
    bearing: float,
    distance_km: float,
) -> tuple[float, float]:
    """Return the endpoint of a spherical great-circle path."""
    earth_radius_km = 6371.0
    angular_distance = distance_km / earth_radius_km
    latitude_1 = np.deg2rad(start_latitude)
    longitude_1 = np.deg2rad(start_longitude)
    bearing_rad = np.deg2rad(bearing)

    latitude_2 = np.arcsin(
        np.sin(latitude_1) * np.cos(angular_distance)
        + np.cos(latitude_1)
        * np.sin(angular_distance)
        * np.cos(bearing_rad)
    )
    longitude_2 = longitude_1 + np.arctan2(
        np.sin(bearing_rad) * np.sin(angular_distance) * np.cos(latitude_1),
        np.cos(angular_distance) - np.sin(latitude_1) * np.sin(latitude_2),
    )
    return float(np.rad2deg(latitude_2)), float(np.rad2deg(longitude_2))


def calculate_atmos_column_wind_profile(radar) -> dict[str, object]:
    """Blend the ATMOS radial-velocity column with a KLOT VAD profile.

    The Py-ART column supplies the observed line-of-sight wind component over
    ATMOS. The all-azimuth KLOT VAD supplies the otherwise unobserved
    cross-beam component. The result supports wind-barb plotting but should be
    described as a column-adjusted single-Doppler estimate, not a standalone
    VAD centered on ATMOS.
    """
    raw_velocity_name = velocity_field(radar)
    if raw_velocity_name is None:
        return {"error": "No Doppler velocity field is available."}

    column_velocity_name = raw_velocity_name
    if DEALIAS_VELOCITY_FOR_COLUMN and raw_velocity_name != "corrected_velocity":
        try:
            corrected_velocity = pyart.correct.dealias_region_based(
                radar,
                vel_field=raw_velocity_name,
            )
            radar.add_field(
                "corrected_velocity",
                corrected_velocity,
                replace_existing=True,
            )
            column_velocity_name = "corrected_velocity"
        except Exception as exc:
            print(
                "WARNING: velocity dealiasing failed; the ATMOS column will use "
                f"original velocity field ({exc}).",
                file=sys.stderr,
            )

    height_levels_m_agl = np.arange(
        WIND_PROFILE_HEIGHT_SPACING_M,
        WIND_PROFILE_MAX_HEIGHT_KM * 1000.0
        + WIND_PROFILE_HEIGHT_SPACING_M,
        WIND_PROFILE_HEIGHT_SPACING_M,
    )
    u_by_sweep: list[np.ndarray] = []
    v_by_sweep: list[np.ndarray] = []

    for sweep_number in range(radar.nsweeps):
        try:
            one_sweep = radar.extract_sweeps([sweep_number])
            if column_velocity_name not in one_sweep.fields:
                continue
            velocity_data = np.ma.asarray(
                one_sweep.fields[column_velocity_name]["data"]
            )
            if np.ma.count(velocity_data) == 0:
                continue

            vad_gatefilter = pyart.filters.GateFilter(one_sweep)
            vad_gatefilter.exclude_invalid(column_velocity_name)
            wind_profile = pyart.retrieve.vad_browning(
                one_sweep,
                column_velocity_name,
                z_want=height_levels_m_agl,
                valid_ray_min=16,
                gatefilter=vad_gatefilter,
            )
        except Exception:
            continue

        u_by_sweep.append(
            np.ma.filled(wind_profile.u_wind, np.nan).astype(float)
        )
        v_by_sweep.append(
            np.ma.filled(wind_profile.v_wind, np.nan).astype(float)
        )

    if not u_by_sweep:
        return {"error": "Py-ART could not retrieve a valid KLOT VAD profile."}

    u_stack = np.asarray(u_by_sweep, dtype=float)
    v_stack = np.asarray(v_by_sweep, dtype=float)
    u_counts = np.sum(np.isfinite(u_stack), axis=0)
    v_counts = np.sum(np.isfinite(v_stack), axis=0)
    u_vad = np.divide(
        np.nansum(u_stack, axis=0),
        u_counts,
        out=np.full(height_levels_m_agl.shape, np.nan),
        where=u_counts > 0,
    )
    v_vad = np.divide(
        np.nansum(v_stack, axis=0),
        v_counts,
        out=np.full(height_levels_m_agl.shape, np.nan),
        where=v_counts > 0,
    )

    try:
        column = pyart.util.column_vertical_profile(
            radar,
            latitude=ATMOS_LAT,
            longitude=ATMOS_LON,
            azimuth_spread=COLUMN_AZIMUTH_SPREAD,
            spatial_spread=COLUMN_SPATIAL_SPREAD,
        )
    except Exception as exc:
        return {"error": f"ATMOS column extraction failed: {exc}"}

    if "height" not in column.coords or column_velocity_name not in column:
        return {
            "error": (
                f"ATMOS column does not contain {column_velocity_name!r} "
                "and a height coordinate."
            )
        }

    column_height_m_asl = np.asarray(column["height"].values, dtype=float)
    column_radial_velocity_m_s = np.ma.filled(
        np.ma.asarray(column[column_velocity_name].values),
        np.nan,
    ).astype(float)
    column_height_m_asl = np.ravel(column_height_m_asl)
    column_radial_velocity_m_s = np.ravel(column_radial_velocity_m_s)

    valid_column = (
        np.isfinite(column_height_m_asl)
        & np.isfinite(column_radial_velocity_m_s)
    )
    if np.count_nonzero(valid_column) < 2:
        return {"error": "ATMOS column contains too few valid velocity levels."}

    radar_altitude_m_asl = float(np.asarray(radar.altitude["data"])[0])
    column_height_m_agl = (
        column_height_m_asl[valid_column] - radar_altitude_m_asl
    )
    column_radial_velocity_m_s = column_radial_velocity_m_s[valid_column]
    order = np.argsort(column_height_m_agl)
    column_height_m_agl = column_height_m_agl[order]
    column_radial_velocity_m_s = column_radial_velocity_m_s[order]

    within_vad_layer = (
        (height_levels_m_agl >= column_height_m_agl[0])
        & (height_levels_m_agl <= column_height_m_agl[-1])
    )
    observed_radial_m_s = np.full(height_levels_m_agl.shape, np.nan)
    observed_radial_m_s[within_vad_layer] = np.interp(
        height_levels_m_agl[within_vad_layer],
        column_height_m_agl,
        column_radial_velocity_m_s,
    )

    radar_latitude = float(np.asarray(radar.latitude["data"])[0])
    radar_longitude = float(np.asarray(radar.longitude["data"])[0])
    atmos_bearing_deg = bearing_degrees(
        radar_latitude,
        radar_longitude,
        ATMOS_LAT,
        ATMOS_LON,
    )
    bearing_rad = np.deg2rad(atmos_bearing_deg)
    radial_unit_east = np.sin(bearing_rad)
    radial_unit_north = np.cos(bearing_rad)
    atmos_range_m = float(
        haversine_km(
            np.asarray([radar_latitude]),
            np.asarray([radar_longitude]),
            ATMOS_LAT,
            ATMOS_LON,
        )[0]
        * 1000.0
    )

    # Approximate the horizontal radial component by removing the beam-angle
    # projection. Vertical air motion cannot be separated by one Doppler radar.
    beam_cosine = atmos_range_m / np.hypot(
        atmos_range_m,
        height_levels_m_agl,
    )
    observed_horizontal_radial_m_s = np.divide(
        observed_radial_m_s,
        beam_cosine,
        out=np.full(height_levels_m_agl.shape, np.nan),
        where=beam_cosine >= 0.5,
    )

    vad_radial_m_s = (
        u_vad * radial_unit_east
        + v_vad * radial_unit_north
    )
    radial_adjustment_m_s = (
        observed_horizontal_radial_m_s - vad_radial_m_s
    )
    u_adjusted = u_vad + radial_adjustment_m_s * radial_unit_east
    v_adjusted = v_vad + radial_adjustment_m_s * radial_unit_north
    valid_adjusted = (
        np.isfinite(u_adjusted)
        & np.isfinite(v_adjusted)
        & np.isfinite(observed_horizontal_radial_m_s)
    )
    u_adjusted[~valid_adjusted] = np.nan
    v_adjusted[~valid_adjusted] = np.nan

    return {
        "height_km": height_levels_m_agl / 1000.0,
        "u_m_s": u_adjusted,
        "v_m_s": v_adjusted,
        "velocity_field": column_velocity_name,
        "sweeps_used": len(u_by_sweep),
        "azimuth_spread": COLUMN_AZIMUTH_SPREAD,
        "spatial_spread": COLUMN_SPATIAL_SPREAD,
    }


def render_scan_frame(
    radar,
    result: dict[str, object],
    street_tiles: GrayscaleOSM,
    column_history: list[tuple[datetime, dict[str, object]]],
    warning_records: list[dict[str, object]],
) -> Image.Image:
    """Render one four-panel radar frame and return it as a PIL image."""
    reflectivity_name = str(result["reflectivity_field"])
    gatefilter = make_gatefilter(radar, reflectivity_name)
    geographic_crs = ccrs.PlateCarree()
    map_projection = ccrs.LambertConformal(
        central_longitude=ATMOS_LON,
        central_latitude=ATMOS_LAT,
        standard_parallels=(40.0, 44.0),
    )

    radar_longitude = float(np.asarray(radar.longitude["data"])[0])
    radar_latitude = float(np.asarray(radar.latitude["data"])[0])
    rhi_azimuth = bearing_degrees(
        radar_latitude,
        radar_longitude,
        ATMOS_LAT,
        ATMOS_LON,
    )
    atmos_range_km = float(
        haversine_km(
            np.asarray([radar_latitude]),
            np.asarray([radar_longitude]),
            ATMOS_LAT,
            ATMOS_LON,
        )[0]
    )
    rhi_start = destination_lat_lon(
        radar_latitude,
        radar_longitude,
        (rhi_azimuth + 180.0) % 360.0,
        RHI_HALF_LENGTH_KM,
    )
    rhi_end = destination_lat_lon(
        radar_latitude,
        radar_longitude,
        rhi_azimuth,
        RHI_HALF_LENGTH_KM,
    )

    # All four panels receive explicit positions. Automatic tight/constrained
    # layouts are avoided because they can crop or resize Cartopy axes.
    figure = plt.figure(figsize=(19, 12.5), facecolor="white")
    map_axes = (
        figure.add_axes([0.025, 0.525, 0.415, 0.375], projection=map_projection),
        figure.add_axes([0.585, 0.525, 0.390, 0.375], projection=map_projection),
    )
    # Leave a larger vertical gutter between the maps and lower panels so
    # map-coordinate labels do not collide with the lower-panel titles.
    rhi_axis = figure.add_axes([0.040, 0.055, 0.385, 0.365])
    column_axis = figure.add_axes([0.575, 0.055, 0.365, 0.365])
    ppi_colorbar_axis = figure.add_axes([0.488, 0.570, 0.014, 0.285])
    rhi_colorbar_axis = figure.add_axes([0.488, 0.095, 0.014, 0.285])
    column_colorbar_axis = figure.add_axes([0.955, 0.095, 0.012, 0.285])

    regional_extent = [
        ATMOS_LON - 0.65,
        ATMOS_LON + 0.65,
        ATMOS_LAT - 0.60,
        ATMOS_LAT + 0.72,
    ]
    zoom_extent = [
        ATMOS_LON - 0.075,
        ATMOS_LON + 0.075,
        ATMOS_LAT - 0.055,
        ATMOS_LAT + 0.055,
    ]
    panel_titles = [
        "a) Chicago metropolitan region",
        "b) ATMOS site zoom",
    ]

    for panel_number, (axis, extent, panel_title) in enumerate(
        zip(map_axes, (regional_extent, zoom_extent), panel_titles)
    ):
        # Add detailed roads beneath the ATMOS zoom. Radar is translucent in
        # this panel so roads and place labels remain visible through echoes.
        if panel_number == 1:
            axis.add_image(street_tiles, 14, zorder=0)

        # Use an independent display object for each axis. This prevents
        # Py-ART's internal current-axis state from affecting the first panel
        # when the second panel is drawn.
        panel_display = pyart.graph.RadarMapDisplay(radar)
        panel_display.plot_ppi_map(
            reflectivity_name,
            sweep=SWEEP,
            ax=axis,
            fig=figure,
            projection=map_projection,
            min_lon=extent[0],
            max_lon=extent[1],
            min_lat=extent[2],
            max_lat=extent[3],
            vmin=MINIMUM_DBZ,
            vmax=70.0,
            cmap="HomeyerRainbow",
            gatefilter=gatefilter,
            colorbar_flag=False,
            title_flag=False,
            embellish=False,
            add_grid_lines=False,
            raster=True,
            alpha=0.86 if panel_number == 1 else 1.0,
        )
        axis.set_extent(extent, crs=geographic_crs)
        # Fill the assigned rectangle instead of letting GeoAxes enlarge or
        # shrink itself to satisfy an automatic fixed-aspect calculation.
        axis.set_aspect("equal", adjustable="datalim")
        add_map_features(axis)

        if panel_number == 0:
            # MetPy supplies these U.S. county geometries, so no external
            # shapefile path is required. Draw them over the radar colors.
            axis.add_feature(
                USCOUNTIES.with_scale("500k"),
                facecolor="none",
                edgecolor="0.15",
                linewidth=0.8,
                zorder=8,
            )
            add_city_labels(
                axis,
                geographic_crs,
                REGIONAL_CITIES,
                marker_size=25,
                font_size=9,
            )
        else:
            add_city_labels(
                axis,
                geographic_crs,
                ZOOM_CITIES,
                marker_size=18,
                font_size=8,
            )

        gridlines = axis.gridlines(
            crs=geographic_crs,
            draw_labels=True,
            linewidth=0.45,
            color="0.45",
            alpha=0.55,
            linestyle="--",
            x_inline=False,
            y_inline=False,
        )
        gridlines.top_labels = False
        gridlines.bottom_labels = "x"
        # Cartopy otherwise rotates some Lambert Conformal longitude labels
        # to follow the projected gridlines.
        gridlines.rotate_labels = False
        if panel_number == 0:
            gridlines.left_labels = False
            gridlines.right_labels = "y"
        else:
            gridlines.left_labels = "y"
            gridlines.right_labels = False
            # Keep the central meridian consistent with the surrounding
            # labels (for example, show 88.00°W rather than 88°W).
            gridlines.xformatter = LongitudeFormatter(
                number_format=".2f",
                auto_hide=False,
            )
        gridlines.xlabel_style = {
            "size": MAP_COORDINATE_FONT_SIZE,
            "rotation": 0,
        }
        gridlines.ylabel_style = {"size": MAP_COORDINATE_FONT_SIZE}
        gridlines.xpadding = 5
        gridlines.ypadding = 2

        axis.scatter(
            ATMOS_LON,
            ATMOS_LAT,
            marker="*",
            s=180,
            color="magenta",
            edgecolor="black",
            linewidth=0.8,
            transform=geographic_crs,
            zorder=10,
            label="ATMOS",
        )
        axis.set_title(panel_title, fontsize=13, fontweight="bold")

        if panel_number == 1:
            axis.text(
                0.99,
                0.01,
                "Road map © OpenStreetMap contributors",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.25",
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
                zorder=21,
            )

    map_axes[0].scatter(
        radar_longitude,
        radar_latitude,
        marker="^",
        s=80,
        color="black",
        transform=geographic_crs,
        zorder=10,
        label=f"{RADAR_SITE} radar",
    )
    map_axes[0].plot(
        [rhi_start[1], rhi_end[1]],
        [rhi_start[0], rhi_end[0]],
        color="#F2F2F2",
        linewidth=2.0,
        linestyle="--",
        path_effects=[
            path_effects.Stroke(linewidth=3.5, foreground="#404040"),
            path_effects.Normal(),
        ],
        transform=geographic_crs,
        zorder=9,
        label=f"RHI transect {rhi_azimuth:.1f}°",
    )
    # Label the actual cross-section endpoints without adding point markers.
    # A is the negative-distance end and B is the end toward ATMOS.
    for endpoint_label, endpoint, lon_offset, lat_offset in (
        ("A", rhi_start, 0.018, 0.018),
        ("B", rhi_end, -0.018, -0.018),
    ):
        map_axes[0].text(
            endpoint[1] + lon_offset,
            endpoint[0] + lat_offset,
            endpoint_label,
            color="#F2F2F2",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            transform=geographic_crs,
            path_effects=[
                path_effects.Stroke(linewidth=3.0, foreground="#404040"),
                path_effects.Normal(),
            ],
            zorder=11,
        )
    map_axes[0].add_patch(
        Rectangle(
            (zoom_extent[0], zoom_extent[2]),
            zoom_extent[1] - zoom_extent[0],
            zoom_extent[3] - zoom_extent[2],
            fill=False,
            edgecolor="magenta",
            linewidth=1.5,
            transform=geographic_crs,
            zorder=9,
        )
    )

    # Show the actual 1 km neighborhood used for the ATMOS rain-rate sample.
    # This is a sampling boundary, not an official property/campus boundary.
    sample_boundary = Geodesic().circle(
        lon=ATMOS_LON,
        lat=ATMOS_LAT,
        radius=SAMPLE_RADIUS_KM * 1000.0,
        n_samples=180,
        endpoint=False,
    )
    map_axes[1].plot(
        sample_boundary[:, 0],
        sample_boundary[:, 1],
        color="magenta",
        linewidth=1.5,
        linestyle="--",
        transform=geographic_crs,
        zorder=11,
        label=f"{SAMPLE_RADIUS_KM:.1f} km sampling area",
    )
    scan_time = result["scan_time_utc"]
    regional_warning_categories = add_warning_overlays(
        map_axes[0],
        geographic_crs,
        warning_records,
        scan_time,
        regional_extent,
    )
    zoom_warning_categories = add_warning_overlays(
        map_axes[1],
        geographic_crs,
        warning_records,
        scan_time,
        zoom_extent,
    )
    add_map_legend(map_axes[0], regional_warning_categories)
    add_map_legend(map_axes[1], zoom_warning_categories)

    # Follow Py-ART's gridded cross-section workflow. This objective analysis
    # interpolates the discrete PPI elevation sweeps onto a regular Cartesian
    # grid before extracting a vertical transect through KLOT toward ATMOS.
    rhi_gatefilter = make_rhi_gatefilter(radar, reflectivity_name)
    horizontal_limit_m = RHI_GRID_HORIZONTAL_LIMIT_KM * 1000.0
    rhi_grid = pyart.map.grid_from_radars(
        (radar,),
        gatefilters=(rhi_gatefilter,),
        grid_shape=(
            RHI_GRID_VERTICAL_POINTS,
            RHI_GRID_HORIZONTAL_POINTS,
            RHI_GRID_HORIZONTAL_POINTS,
        ),
        grid_limits=(
            (0.0, RHI_MAX_HEIGHT_KM * 1000.0),
            (-horizontal_limit_m, horizontal_limit_m),
            (-horizontal_limit_m, horizontal_limit_m),
        ),
        grid_origin=(radar_latitude, radar_longitude),
        fields=[reflectivity_name],
        gridding_algo="map_gates_to_grid",
        weighting_function="Barnes2",
        roi_func="dist_beam",
        min_radius=750.0,
        nb=1.5
    )
    rhi_display = pyart.graph.GridMapDisplay(rhi_grid)
    rhi_display.plot_cross_section(
        reflectivity_name,
        rhi_start,
        rhi_end,
        steps=RHI_CROSS_SECTION_STEPS,
        interp_type="linear",
        ax=rhi_axis,
        fig=figure,
        vmin=RHI_MINIMUM_DBZ,
        vmax=70.0,
        cmap="HomeyerRainbow",
        colorbar_flag=False,
        title_flag=False,
        axislabels_flag=False,
        shading="auto",
        rasterized=True,
    )
    rhi_axis.set_xlim(0.0, RHI_CROSS_SECTION_STEPS - 1)
    rhi_axis.set_ylim(0.0, RHI_MAX_HEIGHT_KM)
    distance_ticks_km = np.linspace(
        -RHI_HALF_LENGTH_KM,
        RHI_HALF_LENGTH_KM,
        7,
    )
    distance_tick_positions = (
        (distance_ticks_km + RHI_HALF_LENGTH_KM)
        / (2.0 * RHI_HALF_LENGTH_KM)
        * (RHI_CROSS_SECTION_STEPS - 1)
    )
    rhi_axis.set_xticks(distance_tick_positions)
    rhi_axis.set_xticklabels([f"{distance:.0f}" for distance in distance_ticks_km])
    rhi_axis.set_xlabel(
        "Signed distance from KLOT along transect (km)",
        fontsize=AXIS_LABEL_FONT_SIZE,
    )
    rhi_axis.set_ylabel("Height above radar (km)", fontsize=AXIS_LABEL_FONT_SIZE)
    rhi_axis.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE)
    rhi_axis.set_title(
        f"c) RHI for KLOT ({rhi_azimuth:.1f}°)",
        fontsize=LOWER_PANEL_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    radar_cross_section_x = 0.5 * (RHI_CROSS_SECTION_STEPS - 1)
    atmos_cross_section_x = (
        (atmos_range_km + RHI_HALF_LENGTH_KM)
        / (2.0 * RHI_HALF_LENGTH_KM)
        * (RHI_CROSS_SECTION_STEPS - 1)
    )
    rhi_axis.axvline(
        radar_cross_section_x,
        color="black",
        linewidth=1.0,
        linestyle=":",
        zorder=10,
    )
    rhi_axis.axvline(
        atmos_cross_section_x,
        color="magenta",
        linewidth=1.5,
        linestyle="--",
        zorder=10,
    )
    rhi_axis.text(
        0.012,
        0.965,
        "A",
        transform=rhi_axis.transAxes,
        ha="left",
        va="top",
        color="#F2F2F2",
        fontsize=12,
        fontweight="bold",
        path_effects=[
            path_effects.Stroke(linewidth=3.0, foreground="#404040"),
            path_effects.Normal(),
        ],
        zorder=11,
    )
    rhi_axis.text(
        0.988,
        0.965,
        "B",
        transform=rhi_axis.transAxes,
        ha="right",
        va="top",
        color="#F2F2F2",
        fontsize=12,
        fontweight="bold",
        path_effects=[
            path_effects.Stroke(linewidth=3.0, foreground="#404040"),
            path_effects.Normal(),
        ],
        zorder=11,
    )
    rhi_axis.text(
        atmos_cross_section_x,
        0.97,
        " ATMOS",
        transform=rhi_axis.get_xaxis_transform(),
        ha="left",
        va="top",
        color="magenta",
        fontweight="bold",
        fontsize=9,
        path_effects=[path_effects.withStroke(linewidth=2.0, foreground="white")],
        zorder=11,
    )
    rhi_axis.grid(True, color="0.75", linestyle="--", linewidth=0.6)

    # Build a time-height wind-barb display from the ATMOS column-adjusted
    # single-Doppler profiles. Barb magnitude remains in knots by convention;
    # color represents the same speed converted to mph.
    column_axis.set_xlim(-0.6, NUMBER_OF_SCANS - 0.4)
    column_axis.set_ylim(0.0, WIND_DISPLAY_MAX_HEIGHT_KFT)
    column_axis.set_xlabel("Scan time (UTC)", fontsize=AXIS_LABEL_FONT_SIZE)
    column_axis.set_ylabel("Height (kft AGL)", fontsize=AXIS_LABEL_FONT_SIZE)
    column_axis.set_title(
        "d) ATMOS Column-Adjusted Wind Barbs - Past 10 Scans",
        fontsize=LOWER_PANEL_TITLE_FONT_SIZE,
        fontweight="bold",
    )
    column_axis.tick_params(axis="both", labelsize=AXIS_TICK_FONT_SIZE)
    column_axis.set_yticks(
        np.arange(0.0, WIND_DISPLAY_MAX_HEIGHT_KFT + 1.0, 1.0)
    )
    column_axis.grid(True, color="0.78", linestyle="--", linewidth=0.6)

    time_positions = np.arange(len(column_history), dtype=float)
    column_axis.set_xticks(time_positions)
    column_axis.set_xticklabels(
        [scan_time.strftime("%H:%M") for scan_time, _ in column_history],
        rotation=0,
        ha="center",
        fontsize=AXIS_TICK_FONT_SIZE,
    )

    display_levels_kft = np.arange(
        1.0,
        WIND_DISPLAY_MAX_HEIGHT_KFT + 0.1,
        1.0,
    )
    wind_speed_norm = Normalize(
        vmin=WIND_SPEED_MIN_MPH,
        vmax=WIND_SPEED_MAX_MPH,
        clip=True,
    )
    wind_speed_cmap = make_wind_speed_colormap()

    for time_index, (_, column_profile) in enumerate(column_history):
        if "error" in column_profile:
            column_axis.text(
                time_index,
                0.45,
                "ND",
                ha="center",
                va="center",
                fontsize=8,
                color="0.35",
            )
            continue

        height_kft = (
            np.asarray(column_profile["height_km"], dtype=float)
            * 3.280839895
        )
        u_knots = (
            np.asarray(column_profile["u_m_s"], dtype=float)
            * M_S_TO_KNOTS
        )
        v_knots = (
            np.asarray(column_profile["v_m_s"], dtype=float)
            * M_S_TO_KNOTS
        )
        valid_wind = (
            np.isfinite(height_kft)
            & np.isfinite(u_knots)
            & np.isfinite(v_knots)
        )
        if np.count_nonzero(valid_wind) < 2:
            column_axis.text(
                time_index,
                0.45,
                "ND",
                ha="center",
                va="center",
                fontsize=8,
                color="0.35",
            )
            continue

        valid_heights = height_kft[valid_wind]
        usable_levels = display_levels_kft[
            (display_levels_kft >= valid_heights[0])
            & (display_levels_kft <= valid_heights[-1])
        ]
        if usable_levels.size == 0:
            continue

        interpolated_u_knots = np.interp(
            usable_levels,
            valid_heights,
            u_knots[valid_wind],
        )
        interpolated_v_knots = np.interp(
            usable_levels,
            valid_heights,
            v_knots[valid_wind],
        )
        interpolated_speed_mph = (
            np.hypot(interpolated_u_knots, interpolated_v_knots)
            * KNOT_TO_MPH
        )

        column_axis.barbs(
            np.full(usable_levels.shape, float(time_index)),
            usable_levels,
            interpolated_u_knots,
            interpolated_v_knots,
            interpolated_speed_mph,
            length=5.2,
            linewidth=0.75,
            pivot="middle",
            barb_increments={"half": 5, "full": 10, "flag": 50},
            cmap=wind_speed_cmap,
            norm=wind_speed_norm,
            zorder=3,
        )

    # Fixed, scan-independent colorbars preserve the existing PPI scale while
    # allowing the gridded RHI to show weaker echo down to -10 dBZ.
    ppi_colorbar_mappable = ScalarMappable(
        norm=Normalize(vmin=MINIMUM_DBZ, vmax=70.0),
        cmap=plt.get_cmap("HomeyerRainbow"),
    )
    ppi_colorbar_mappable.set_array([])
    ppi_colorbar = figure.colorbar(
        ppi_colorbar_mappable,
        cax=ppi_colorbar_axis,
        ticks=[10, 20, 30, 40, 50, 60, 70],
    )
    ppi_colorbar.set_label(
        "Reflectivity (dBZ)",
        fontsize=COLORBAR_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    ppi_colorbar_axis.tick_params(labelsize=AXIS_TICK_FONT_SIZE)

    rhi_colorbar_mappable = ScalarMappable(
        norm=Normalize(vmin=RHI_MINIMUM_DBZ, vmax=70.0),
        cmap=plt.get_cmap("HomeyerRainbow"),
    )
    rhi_colorbar_mappable.set_array([])
    rhi_colorbar = figure.colorbar(
        rhi_colorbar_mappable,
        cax=rhi_colorbar_axis,
        ticks=[-10, 0, 10, 20, 30, 40, 50, 60, 70],
    )
    rhi_colorbar.set_label(
        "RHI reflectivity (dBZ)",
        fontsize=COLORBAR_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    rhi_colorbar_axis.tick_params(labelsize=AXIS_TICK_FONT_SIZE)
    # Keep this label in the center gutter and away from the column y-axis.
    rhi_colorbar_axis.yaxis.set_ticks_position("left")
    rhi_colorbar_axis.yaxis.set_label_position("left")

    column_colorbar_mappable = ScalarMappable(
        norm=wind_speed_norm,
        cmap=wind_speed_cmap,
    )
    column_colorbar_mappable.set_array([])
    column_colorbar = figure.colorbar(
        column_colorbar_mappable,
        cax=column_colorbar_axis,
        ticks=WIND_SPEED_COLORBAR_TICKS_MPH,
        extend="max",
    )
    column_colorbar.set_label(
        "Wind speed (mph)",
        fontsize=COLORBAR_LABEL_FONT_SIZE,
        fontweight="bold",
    )
    column_colorbar_axis.tick_params(labelsize=AXIS_TICK_FONT_SIZE)

    scan_time = result["scan_time_utc"]
    scan_time_text = scan_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    scan_age_minutes = (
        datetime.now(timezone.utc) - scan_time
    ).total_seconds() / 60.0

    figure.suptitle(
        f"NWS {RADAR_SITE} Radar / ATMOS Site Analysis — {scan_time_text}",
        fontsize=22,
        fontweight="bold",
        y=0.945,
    )
    map_axes[1].text(
        0.02,
        0.02,
        "KLOT radar estimate\n"
        f"Rain rate: {float(result['rain_rate_mm_h']):.2f} mm h$^{{-1}}$\n"
        f"Median reflectivity: {float(result['median_reflectivity_dbz']):.1f} dBZ\n"
        f"Gates used: {int(result['gates_used'])}\n"
        f"Sampling radius: {SAMPLE_RADIUS_KM:.1f} km\n"
        f"Scan age at plotting: {scan_age_minutes:.1f} min",
        transform=map_axes[1].transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
        zorder=20,
    )

    # Reassert positions after Py-ART, Cartopy, and the colorbar add artists.
    map_axes[0].set_position([0.025, 0.525, 0.415, 0.375])
    map_axes[1].set_position([0.585, 0.525, 0.390, 0.375])
    rhi_axis.set_position([0.040, 0.055, 0.385, 0.365])
    column_axis.set_position([0.575, 0.055, 0.365, 0.365])
    ppi_colorbar_axis.set_position([0.488, 0.570, 0.014, 0.285])
    rhi_colorbar_axis.set_position([0.488, 0.095, 0.014, 0.285])
    column_colorbar_axis.set_position([0.955, 0.095, 0.012, 0.285])
    for axis in map_axes:
        axis.set_aspect("equal", adjustable="datalim")

    # Render into memory. No intermediate frame PNGs are written to disk.
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=GIF_DPI, facecolor="white")
    plt.close(figure)
    buffer.seek(0)
    frame = Image.open(buffer).convert("RGB")
    frame.load()
    buffer.close()
    return frame


def freeze_completed_column_panel(frames: list[Image.Image]) -> None:
    """Copy the completed ten-scan ATMOS column panel into every GIF frame.

    The final rendered frame contains the full column time history. Reusing
    that panel avoids re-reading the radar volumes or retaining ten large
    Py-ART Radar objects in memory, while leaving the other three panels
    animated.
    """
    if len(frames) < 2:
        return

    frame_width, frame_height = frames[-1].size
    # Crop in image coordinates (origin at upper left). This includes the
    # column title and both axes, but stops to the right of the central RHI
    # colorbar.
    column_panel_box = (
        int(round(0.505 * frame_width)),
        int(round(0.495 * frame_height)),
        frame_width,
        frame_height,
    )
    completed_column_panel = frames[-1].crop(column_panel_box)
    try:
        for frame in frames[:-1]:
            frame.paste(completed_column_panel, column_panel_box[:2])
    finally:
        completed_column_panel.close()


def main() -> int:
    """Remotely process the latest KLOT volumes and save one looping GIF."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Searching NOAA's public archive for the newest "
        f"{NUMBER_OF_SCANS} {RADAR_SITE} scans..."
    )
    try:
        radar_objects = find_latest_nexrad_objects(NUMBER_OF_SCANS)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not radar_objects:
        print("ERROR: no radar scans were available for the GIF.", file=sys.stderr)
        return 1

    scan_times = [scan_time for _, scan_time in radar_objects]
    try:
        warning_records = fetch_nws_warning_records(scan_times)
        print("NWS warning overlay data acquired.")
    except Exception as exc:
        warning_records = []
        print(
            "WARNING: NWS warning polygons could not be retrieved; "
            f"continuing without them ({exc}).",
            file=sys.stderr,
        )

    gif_path = OUTPUT_DIR / "last_10_klot_atmos_four_panel.gif"
    street_tiles = GrayscaleOSM()
    frames: list[Image.Image] = []
    results: list[dict[str, object]] = []
    column_history: list[tuple[datetime, dict[str, object]]] = []
    frame_count = len(radar_objects)

    for frame_number, (object_name, filename_scan_time) in enumerate(
        radar_objects, start=1
    ):
        radar_source = f"s3://{object_name}"
        print(
            f"[{frame_number:>2}/{frame_count}] Reading remotely: "
            f"{Path(object_name).name}"
        )
        try:
            radar = pyart.io.read_nexrad_archive(
                radar_source,
                storage_options={"anon": True},
            )
            result = estimate_atmos_rain_rate(radar, radar_source)
            result["archive_filename_time"] = filename_scan_time
            column_profile = calculate_atmos_column_wind_profile(radar)
            column_history.append((result["scan_time_utc"], column_profile))
            frames.append(
                render_scan_frame(
                    radar,
                    result,
                    street_tiles,
                    column_history,
                    warning_records,
                )
            )
            results.append(result)
        except Exception as exc:
            print(
                f"ERROR while processing {Path(object_name).name}: {exc}",
                file=sys.stderr,
            )
            for frame in frames:
                frame.close()
            return 1

    durations = [FRAME_DURATION_MS] * len(frames)
    durations[-1] = FINAL_FRAME_DURATION_MS
    # The top maps and pseudo-RHI animate by scan; the ATMOS column remains a
    # still ten-scan time-height summary in every frame.
    freeze_completed_column_panel(frames)
    # Use the first frame's adaptive palette as one global palette for the
    # entire GIF. This prevents frame-by-frame palette changes from altering
    # the appearance of the fixed reflectivity colorbar.
    palette_reference = frames[0].convert(
        "P",
        palette=Image.Palette.ADAPTIVE,
        colors=256,
    )
    gif_frames = [
        frame.quantize(
            palette=palette_reference,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    try:
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=1,
        )
    except Exception as exc:
        print(f"ERROR while saving GIF: {exc}", file=sys.stderr)
        return 1
    finally:
        palette_reference.close()
        for frame in gif_frames:
            frame.close()
        for frame in frames:
            frame.close()

    first_time = results[0]["scan_time_utc"]
    final_time = results[-1]["scan_time_utc"]
    final_age_minutes = (
        datetime.now(timezone.utc) - final_time
    ).total_seconds() / 60.0

    print("\nKLOT/ATMOS four-panel radar animation complete")
    print(f"  Frames:                {len(results)}")
    print(f"  First scan:            {first_time:%Y-%m-%d %H:%M:%S UTC}")
    print(f"  Final scan:            {final_time:%Y-%m-%d %H:%M:%S UTC}")
    print(f"  Latest scan age:       {final_age_minutes:.1f} minutes")
    print(f"  GIF:                   {gif_path}")
    print("  RHI:                   Gridded pseudo-RHI through KLOT toward ATMOS")
    print(
        "  Wind profile:          ATMOS radial-velocity column blended with "
        "the KLOT VAD cross-beam component"
    )
    print("  Warnings:              NWS polygons matched to each scan time")
    print("  Note: each frame shows an instantaneous rain rate, not accumulation.")

    if final_age_minutes > 30.0:
        print(
            "  WARNING: the newest available scan is more than 30 minutes old; "
            "KLOT or the public archive may be delayed."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
