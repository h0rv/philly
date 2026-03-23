from __future__ import annotations

import asyncio
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from zipfile import ZipFile

from philly import Philly

ROOT = Path(__file__).parent
OUT_PATH = ROOT / "data" / "transit.json"
GTFS_PUBLIC_URL = "https://www3.septa.org/developer/gtfs_public.zip"
TRAIN_VIEW_URL = "https://www3.septa.org/api/TrainView/index.php"
METRO_RESOURCE_NAMES = [
    "Market-Frankford Line Trips API",
    "Broad Street Line Trips API (All Services)",
    "Norristown High Speed Line Trips API",
]

REGIONAL_BOUNDS = {
    "west": -75.95,
    "south": 39.55,
    "east": -74.75,
    "north": 40.45,
}

PHILLY_CAMERA = {
    "longitude": -75.1652,
    "latitude": 39.9526,
    "zoom": 11.9,
    "pitch": 54,
    "bearing": 16,
}

MODE_SPEEDS_MPS = {
    "bus": 7.5,
    "trolley": 6.5,
    "metro": 11.0,
    "regional_rail": 18.0,
}

MODE_HEIGHTS_METERS = {
    "bus": 18,
    "trolley": 18,
    "metro": 22,
    "regional_rail": 28,
}

MODE_WIDTHS_METERS = {
    "bus": 8,
    "trolley": 8,
    "metro": 9,
    "regional_rail": 10,
}

MODE_LENGTHS_METERS = {
    "bus": 20,
    "trolley": 22,
    "metro": 28,
    "regional_rail": 34,
}

MODE_TOLERANCE_METERS = {
    "bus": 45,
    "trolley": 35,
    "metro": 30,
    "regional_rail": 80,
}

GTFS_STATUS = {
    0: "IN_TRANSIT_TO",
    1: "STOPPED_AT",
    2: "INCOMING_AT",
}

GTFS_OCCUPANCY = {
    0: None,
    1: "empty",
    2: "many seats available",
    3: "few seats available",
    4: "standing room only",
    5: "crushed standing room only",
    6: "full",
    7: "not accepting passengers",
}


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "philly-transit-mvp/0.1"})
    with urlopen(request, timeout=120) as response:
        return response.read()


def fetch_json(url: str) -> Any:
    return json.loads(fetch_bytes(url).decode("utf-8"))


def parse_csv_from_zip(zip_file: ZipFile, filename: str) -> list[dict[str, str]]:
    with zip_file.open(filename) as handle:
        wrapper = TextIOWrapper(handle, encoding="utf-8-sig")
        return list(csv.DictReader(wrapper))


def in_bounds(lat: float, lng: float) -> bool:
    return (
        REGIONAL_BOUNDS["south"] <= lat <= REGIONAL_BOUNDS["north"]
        and REGIONAL_BOUNDS["west"] <= lng <= REGIONAL_BOUNDS["east"]
    )


def hex_to_rgb(color: str | None) -> list[int]:
    value = (color or "666666").strip().lstrip("#")
    if len(value) != 6:
        value = "666666"
    return [int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)]


def clean_token(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "nan", "tbd"}:
        return fallback
    return text


def normalize_feed_timestamp(transit_date: Any, raw_timestamp: Any) -> int | None:
    if raw_timestamp in (None, ""):
        return None
    try:
        value = int(float(raw_timestamp))
    except (TypeError, ValueError):
        return None
    if value > 1_000_000_000:
        return value
    if not transit_date:
        return None
    try:
        base = datetime.strptime(str(transit_date), "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    return int(base.timestamp()) + value


def parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def route_mode(route: dict[str, str] | None, route_id: str) -> str:
    route_type = (route or {}).get("route_type")
    if route_type == "0":
        return "trolley"
    if route_type == "1":
        return "metro"
    if route_type == "2":
        return "regional_rail"
    if route_id.startswith("T"):
        return "trolley"
    return "bus"


def mode_label(mode: str) -> str:
    return {
        "bus": "Bus",
        "trolley": "Trolley",
        "metro": "Metro",
        "regional_rail": "Regional Rail",
    }.get(mode, mode.replace("_", " ").title())


def default_speed(mode: str, status_name: str | None) -> float:
    base = MODE_SPEEDS_MPS.get(mode, 7.0)
    if status_name == "STOPPED_AT":
        return base * 0.25
    if status_name == "INCOMING_AT":
        return base * 0.55
    return base


def project(lng: float, lat: float, ref_lat: float) -> tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lng = 111_320.0 * math.cos(math.radians(ref_lat))
    return lng * meters_per_deg_lng, lat * meters_per_deg_lat


def distance_m(a: tuple[float, float], b: tuple[float, float], ref_lat: float) -> float:
    ax, ay = project(a[0], a[1], ref_lat)
    bx, by = project(b[0], b[1], ref_lat)
    return math.hypot(bx - ax, by - ay)


def perpendicular_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
    ref_lat: float,
) -> float:
    px, py = project(point[0], point[1], ref_lat)
    sx, sy = project(start[0], start[1], ref_lat)
    ex, ey = project(end[0], end[1], ref_lat)
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def simplify_polyline(
    points: list[tuple[float, float]], tolerance_m: float
) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points

    ref_lat = sum(lat for _, lat in points) / len(points)

    def rdp(segment: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(segment) <= 2:
            return segment
        start = segment[0]
        end = segment[-1]
        max_dist = -1.0
        index = -1
        for i in range(1, len(segment) - 1):
            dist = perpendicular_distance(segment[i], start, end, ref_lat)
            if dist > max_dist:
                max_dist = dist
                index = i
        if max_dist <= tolerance_m:
            return [start, end]
        left = rdp(segment[: index + 1])
        right = rdp(segment[index:])
        return left[:-1] + right

    simplified = rdp(points)
    deduped: list[tuple[float, float]] = []
    for point in simplified:
        if not deduped or point != deduped[-1]:
            deduped.append(point)
    return deduped


def nearest_progress(coords: list[list[float]], point: list[float]) -> float:
    if len(coords) < 2:
        return 0.0

    ref_lat = point[1]
    px, py = project(point[0], point[1], ref_lat)
    best_distance = float("inf")
    best_progress = 0.0
    progress_so_far = 0.0

    for start, end in zip(coords, coords[1:]):
        sx, sy = project(start[0], start[1], ref_lat)
        ex, ey = project(end[0], end[1], ref_lat)
        dx = ex - sx
        dy = ey - sy
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            continue
        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        dist = math.hypot(px - proj_x, py - proj_y)
        if dist < best_distance:
            best_distance = dist
            best_progress = progress_so_far + seg_len * t
        progress_so_far += seg_len

    return best_progress


def polyline_length(coords: list[list[float]]) -> float:
    if len(coords) < 2:
        return 0.0
    ref_lat = sum(lat for _, lat in coords) / len(coords)
    total = 0.0
    for start, end in zip(coords, coords[1:]):
        total += distance_m((start[0], start[1]), (end[0], end[1]), ref_lat)
    return total


def shape_is_loop(coords: list[list[float]]) -> bool:
    if len(coords) < 3:
        return False
    ref_lat = sum(lat for _, lat in coords) / len(coords)
    start = (coords[0][0], coords[0][1])
    end = (coords[-1][0], coords[-1][1])
    return distance_m(start, end, ref_lat) <= 120.0


def load_gtfs_tables() -> dict[str, Any]:
    outer = ZipFile(BytesIO(fetch_bytes(GTFS_PUBLIC_URL)))
    result: dict[str, Any] = {}

    for bundle_name, prefix in (("google_bus.zip", "bus"), ("google_rail.zip", "rail")):
        inner = ZipFile(BytesIO(outer.read(bundle_name)))
        routes = parse_csv_from_zip(inner, "routes.txt")
        trips = parse_csv_from_zip(inner, "trips.txt")
        stops = parse_csv_from_zip(inner, "stops.txt")
        shapes = parse_csv_from_zip(inner, "shapes.txt")
        result[prefix] = {
            "routes": routes,
            "trips": trips,
            "stops": stops,
            "shapes": shapes,
        }

    return result


def build_shape_map(
    shape_rows: list[dict[str, str]],
    route_lookup: dict[str, dict[str, str]],
    shape_route_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[int, float, float]]] = {}
    for row in shape_rows:
        grouped.setdefault(row["shape_id"], []).append(
            (
                int(float(row["shape_pt_sequence"])),
                float(row["shape_pt_lon"]),
                float(row["shape_pt_lat"]),
            )
        )

    shape_map: dict[str, dict[str, Any]] = {}
    for shape_id, points in grouped.items():
        route = route_lookup.get(shape_route_map.get(shape_id, ""))
        mode = route_mode(route, shape_route_map.get(shape_id, ""))
        tolerance = MODE_TOLERANCE_METERS.get(mode, 45)
        ordered = sorted(points, key=lambda item: item[0])
        coords = [[lng, lat] for _, lng, lat in ordered]
        simplified = simplify_polyline([(lng, lat) for lng, lat in coords], tolerance)
        shape_coords = [[lng, lat] for lng, lat in simplified]
        if len(shape_coords) < 2:
            continue
        shape_map[shape_id] = {
            "coords": shape_coords,
            "lengthM": round(polyline_length(shape_coords), 1),
            "loop": shape_is_loop(shape_coords),
            "routeId": shape_route_map.get(shape_id),
            "mode": mode,
        }

    return shape_map


async def build() -> dict[str, Any]:
    gtfs = load_gtfs_tables()

    bus_routes = {row["route_id"]: row for row in gtfs["bus"]["routes"]}
    rail_routes = {row["route_id"]: row for row in gtfs["rail"]["routes"]}
    all_routes = {**bus_routes, **rail_routes}

    bus_trips = {row["trip_id"]: row for row in gtfs["bus"]["trips"]}
    rail_trips = {row["trip_id"]: row for row in gtfs["rail"]["trips"]}
    all_trips = {**bus_trips, **rail_trips}

    bus_stops = {row["stop_id"]: row for row in gtfs["bus"]["stops"]}
    rail_stops = {row["stop_id"]: row for row in gtfs["rail"]["stops"]}
    all_stops = {**bus_stops, **rail_stops}

    shape_route_map: dict[str, str] = {}
    for trip in gtfs["bus"]["trips"] + gtfs["rail"]["trips"]:
        shape_id = trip.get("shape_id")
        route_id = trip.get("route_id")
        if shape_id and route_id and shape_id not in shape_route_map:
            shape_route_map[shape_id] = route_id

    shape_map = build_shape_map(
        gtfs["bus"]["shapes"] + gtfs["rail"]["shapes"],
        all_routes,
        shape_route_map,
    )

    phl = Philly(cache=False)
    bus_rt = await phl.load(
        "SEPTA GTFS Real-time Alerts and Updates",
        resource_name="Bus Vehicle Position Updates in GTFS-RT format",
    )
    rail_rt = await phl.load(
        "SEPTA GTFS Real-time Alerts and Updates",
        resource_name="Regional Rail Vehicle Position Updates in GTFS-RT format",
    )
    metro_trips: list[dict[str, Any]] = []
    for resource_name in METRO_RESOURCE_NAMES:
        resource_data = await phl.load(
            "SEPTA Metro Real-time Trips",
            resource_name=resource_name,
        )
        if isinstance(resource_data, list):
            metro_trips.extend(resource_data)
    train_view = fetch_json(TRAIN_VIEW_URL)
    train_view_by_number = {str(item.get("trainno")): item for item in train_view}

    vehicles: list[dict[str, Any]] = []
    used_shapes: set[str] = set()
    used_routes: set[str] = set()
    timestamps: list[int] = []

    for feed in (bus_rt, rail_rt):
        for entity in feed.entity:
            vehicle = entity.vehicle
            trip_id = vehicle.trip.trip_id
            route_id = vehicle.trip.route_id
            trip = all_trips.get(trip_id)
            route = all_routes.get(route_id)
            shape_id = trip.get("shape_id") if trip else None
            if not shape_id or shape_id not in shape_map:
                continue

            stop_id = vehicle.stop_id or ""
            stop = all_stops.get(stop_id, {})

            lat = float(vehicle.position.latitude)
            lng = float(vehicle.position.longitude)
            position_source = "gps"
            if lat == 0 or lng == 0 or not in_bounds(lat, lng):
                stop_lat = stop.get("stop_lat")
                stop_lng = stop.get("stop_lon")
                if stop_lat and stop_lng:
                    lat = float(stop_lat)
                    lng = float(stop_lng)
                    if in_bounds(lat, lng):
                        position_source = "stop_inferred"
                    else:
                        continue
                else:
                    continue

            status_name = GTFS_STATUS.get(int(vehicle.current_status))
            mode = route_mode(route, route_id)
            if mode == "metro":
                continue
            timestamp = int(vehicle.timestamp) if vehicle.timestamp else 0
            if timestamp:
                timestamps.append(timestamp)

            progress_m = nearest_progress(shape_map[shape_id]["coords"], [lng, lat])
            used_shapes.add(shape_id)
            used_routes.add(route_id)

            train_info = train_view_by_number.get(
                str(vehicle.vehicle.label or entity.id or ""), {}
            )
            occupancy = GTFS_OCCUPANCY.get(int(vehicle.occupancy_status))
            delay = train_info.get("late")
            try:
                delay_minutes = int(delay) if delay not in (None, "") else None
            except ValueError:
                delay_minutes = None

            vehicle_label = clean_token(
                vehicle.vehicle.label or vehicle.vehicle.id or entity.id, str(trip_id)
            )
            unique_vehicle_id = "::".join(
                [
                    mode,
                    clean_token(route_id, "unknown-route"),
                    clean_token(trip_id, "unknown-trip"),
                    vehicle_label,
                    clean_token(entity.id or vehicle.vehicle.id, "unknown-entity"),
                ]
            )

            vehicles.append(
                {
                    "id": unique_vehicle_id,
                    "label": vehicle_label,
                    "agency": "SEPTA",
                    "mode": mode,
                    "modeLabel": mode_label(mode),
                    "routeId": route_id,
                    "routeShortName": (route or {}).get("route_short_name") or route_id,
                    "routeLongName": (route or {}).get("route_long_name") or route_id,
                    "shapeId": shape_id,
                    "progressM": round(progress_m, 1),
                    "speedMps": round(default_speed(mode, status_name), 2),
                    "lengthM": MODE_LENGTHS_METERS[mode],
                    "widthM": MODE_WIDTHS_METERS[mode],
                    "heightM": MODE_HEIGHTS_METERS[mode],
                    "timestamp": timestamp,
                    "currentStatus": status_name,
                    "bearing": round(float(vehicle.position.bearing), 1),
                    "positionSource": position_source,
                    "currentStop": stop.get("stop_name")
                    or train_info.get("currentstop")
                    or None,
                    "nextStop": train_info.get("nextstop") or None,
                    "destination": train_info.get("dest") or None,
                    "delayMinutes": delay_minutes,
                    "service": train_info.get("service") or None,
                    "track": train_info.get("TRACK") or None,
                    "source": train_info.get("SOURCE") or None,
                    "consist": train_info.get("consist") or None,
                    "occupancy": occupancy,
                    "color": hex_to_rgb((route or {}).get("route_color")),
                    "textColor": hex_to_rgb(
                        (route or {}).get("route_text_color") or "FFFFFF"
                    ),
                }
            )

    seen_metro_keys: set[tuple[str, str]] = set()
    for trip_data in metro_trips:
        route_id = clean_token(trip_data.get("route_id"), "")
        trip_id = clean_token(trip_data.get("trip_id"), "")
        if not route_id or not trip_id or (route_id, trip_id) in seen_metro_keys:
            continue
        seen_metro_keys.add((route_id, trip_id))

        trip = all_trips.get(trip_id)
        route = all_routes.get(route_id)
        shape_id = trip.get("shape_id") if trip else None
        if not shape_id or shape_id not in shape_map:
            continue

        mode = route_mode(route, route_id)
        if mode != "metro":
            continue

        next_stop_id = clean_token(trip_data.get("next_stop_id"), "")
        next_stop = all_stops.get(next_stop_id, {})

        lat = parse_float(trip_data.get("lat"))
        lng = parse_float(trip_data.get("lon"))
        bearing_value = parse_float(trip_data.get("heading")) or 0.0
        progress_m: float | None = None
        position_source = "gps"

        if lat is not None and lng is not None and in_bounds(lat, lng):
            progress_m = nearest_progress(shape_map[shape_id]["coords"], [lng, lat])
        elif next_stop.get("stop_lat") and next_stop.get("stop_lon"):
            lat = float(next_stop["stop_lat"])
            lng = float(next_stop["stop_lon"])
            progress_m = nearest_progress(shape_map[shape_id]["coords"], [lng, lat])
            position_source = "stop_inferred"
        else:
            completion = parse_float(trip_data.get("trip_completion"))
            shape_length = float(shape_map[shape_id]["lengthM"])
            if completion is None or completion < 0 or completion > 100:
                continue
            progress_m = max(
                0.0, min(shape_length, shape_length * (completion / 100.0))
            )
            position_source = "progress_inferred"

        timestamp = normalize_feed_timestamp(
            trip_data.get("transit_date"), trip_data.get("timestamp")
        )
        if not timestamp:
            continue
        timestamps.append(timestamp)
        used_shapes.add(shape_id)
        used_routes.add(route_id)

        delay_raw = parse_float(trip_data.get("delay"))
        delay_minutes = None
        if delay_raw is not None and delay_raw < 900:
            delay_minutes = int(round(delay_raw))

        vehicle_label = clean_token(trip_data.get("vehicle_id"), trip_id)
        unique_vehicle_id = "::".join(
            [
                mode,
                clean_token(route_id, "unknown-route"),
                clean_token(trip_id, "unknown-trip"),
                vehicle_label,
                clean_token(trip_data.get("block_id"), "unknown-block"),
            ]
        )

        vehicles.append(
            {
                "id": unique_vehicle_id,
                "label": vehicle_label,
                "agency": "SEPTA",
                "mode": mode,
                "modeLabel": mode_label(mode),
                "routeId": route_id,
                "routeShortName": (route or {}).get("route_short_name") or route_id,
                "routeLongName": (route or {}).get("route_long_name") or route_id,
                "shapeId": shape_id,
                "progressM": round(progress_m or 0.0, 1),
                "speedMps": round(default_speed(mode, None), 2),
                "lengthM": MODE_LENGTHS_METERS[mode],
                "widthM": MODE_WIDTHS_METERS[mode],
                "heightM": MODE_HEIGHTS_METERS[mode],
                "timestamp": timestamp,
                "currentStatus": clean_token(trip_data.get("status"), "Unknown"),
                "bearing": round(bearing_value, 1),
                "positionSource": position_source,
                "currentStop": None,
                "nextStop": trip_data.get("next_stop_name")
                or next_stop.get("stop_name")
                or None,
                "destination": trip_data.get("trip_headsign") or None,
                "delayMinutes": delay_minutes,
                "service": None,
                "track": None,
                "source": None,
                "consist": None,
                "occupancy": None,
                "color": hex_to_rgb((route or {}).get("route_color")),
                "textColor": hex_to_rgb(
                    (route or {}).get("route_text_color") or "FFFFFF"
                ),
            }
        )

    routes_payload = {
        route_id: {
            "routeId": route_id,
            "shortName": route.get("route_short_name") or route_id,
            "longName": route.get("route_long_name") or route_id,
            "mode": route_mode(route, route_id),
            "color": hex_to_rgb(route.get("route_color")),
            "textColor": hex_to_rgb(route.get("route_text_color") or "FFFFFF"),
        }
        for route_id, route in all_routes.items()
        if route_id in used_routes
    }

    shapes_payload = {
        shape_id: {
            **shape_map[shape_id],
            "coords": shape_map[shape_id]["coords"],
        }
        for shape_id in used_shapes
    }

    generated_at = max(timestamps) if timestamps else 0
    mode_counts = Counter(vehicle["mode"] for vehicle in vehicles)
    metro_active_counts = Counter(
        clean_token(item.get("route_id"), "")
        for item in metro_trips
        if item.get("route_id")
    )
    metro_located_counts = Counter(
        vehicle["routeId"] for vehicle in vehicles if vehicle["mode"] == "metro"
    )
    metro_coverage = {
        route_id: {
            "routeId": route_id,
            "routeShortName": (all_routes.get(route_id) or {}).get("route_short_name")
            or route_id,
            "routeLongName": (all_routes.get(route_id) or {}).get("route_long_name")
            or route_id,
            "activeTrips": metro_active_counts.get(route_id, 0),
            "locatedTrips": metro_located_counts.get(route_id, 0),
            "missingTrips": max(
                0,
                metro_active_counts.get(route_id, 0)
                - metro_located_counts.get(route_id, 0),
            ),
        }
        for route_id in sorted(metro_active_counts)
    }

    return {
        "generatedAt": generated_at,
        "generatedAtIso": (
            datetime.fromtimestamp(generated_at, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
            if generated_at
            else None
        ),
        "camera": PHILLY_CAMERA,
        "bounds": REGIONAL_BOUNDS,
        "modes": {
            mode: {
                "label": mode_label(mode),
                "count": mode_counts.get(mode, 0),
                "speedMps": MODE_SPEEDS_MPS.get(mode),
            }
            for mode in ("bus", "trolley", "metro", "regional_rail")
        },
        "stats": {
            "vehicleCount": len(vehicles),
            "shapeCount": len(shapes_payload),
            "routeCount": len(routes_payload),
        },
        "coverage": {
            "metro": metro_coverage,
        },
        "routes": routes_payload,
        "shapes": shapes_payload,
        "vehicles": vehicles,
        "notes": [
            "Snapshot sourced from SEPTA GTFS-RT, SEPTA Metro trips, and TrainView enrichment.",
            "Replay is generated client-side by moving each vehicle along its GTFS shape with route-clamped dead reckoning.",
            "Market-Frankford and Broad Street service may be active without public GPS coordinates; that is a feed limitation, not a frontend bug.",
            "If SEPTA omits a coordinate but provides a stop, the vehicle is anchored to that stop and marked as inferred in the UI.",
            "This MVP ships as a static exploration; the live websocket/history pipeline from the spec is the next step.",
        ],
    }


def main() -> None:
    payload = asyncio.run(build())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"Wrote {OUT_PATH}")
    print(
        f"{payload['stats']['vehicleCount']} vehicles • {payload['stats']['shapeCount']} shapes • {payload['stats']['routeCount']} routes"
    )


if __name__ == "__main__":
    main()
