"""Build the first real block changelog pipeline.

Phase 1 MVP sources:
- Census Blocks
- 311 Service and Information Requests
- Crime Incidents

Outputs:
- data/raw/*.parquet
- data/silver/events_311.parquet
- data/silver/events_crime.parquet
- data/gold/block_log.parquet
- data/gold/block_summary.parquet
- data/gold/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd

from philly import Philly

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
APP_DATA_DIR = BASE_DIR / "app" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-date",
        default="2025-01-01",
        help="Inclusive start date for events (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Inclusive end date for events (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse existing raw parquet files if present.",
    )
    parser.add_argument(
        "--export-app-data",
        action="store_true",
        help="Also export a small JSON sample into app/data for frontend wiring.",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [RAW_DIR, SILVER_DIR, GOLD_DIR, APP_DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def clean_id(series: pd.Series, width: int | None = None) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    if width is not None:
        cleaned = cleaned.str.zfill(width)
    return cleaned


def clean_text(series: pd.Series | None, fallback: str | None = None) -> pd.Series:
    if series is None:
        base = pd.Series(pd.NA, index=pd.RangeIndex(0), dtype="string")
    else:
        base = series.astype("string").str.strip()
        base = base.replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    if fallback is not None:
        return base.fillna(fallback)
    return base


def make_description(*parts: pd.Series | None) -> pd.Series:
    normalized_parts: list[pd.Series] = []
    reference_index: pd.Index | None = None

    for part in parts:
        if part is None:
            continue
        cleaned = clean_text(part)
        if reference_index is None:
            reference_index = cleaned.index
        normalized_parts.append(cleaned)

    if reference_index is None:
        return pd.Series(dtype="string")

    result = pd.Series("", index=reference_index, dtype="string")
    for cleaned in normalized_parts:
        cleaned = cleaned.reindex(reference_index)
        has_text = cleaned.notna()
        needs_sep = result.ne("") & has_text
        result = result.mask(needs_sep, result + " · " + cleaned.fillna(""))
        result = result.mask(result.eq("") & has_text, cleaned.fillna(""))

    return result.replace({"": pd.NA})


async def fetch_census_blocks(phl: Philly, refresh: bool) -> gpd.GeoDataFrame:
    raw_path = RAW_DIR / "census_blocks_2010.parquet"
    if raw_path.exists() and not refresh:
        print(f"Loading cached blocks from {raw_path}")
        gdf = gpd.read_parquet(raw_path)
    else:
        print("Fetching Census Blocks (2010 GeoJSON)...")
        payload = await phl.load(
            "Census Blocks",
            resource_name="Census Blocks - 2010 (GeoJSON)",
            use_cache=True,
        )
        assert isinstance(payload, dict) and "features" in payload
        gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
        gdf.to_parquet(raw_path, index=False)
        print(f"Saved raw blocks to {raw_path}")

    gdf.columns = [column.lstrip("\ufeff") for column in gdf.columns]
    gdf["block_id"] = clean_id(gdf["GEOID10"], width=15)
    gdf["tract_id"] = gdf["block_id"].str.slice(0, 11)
    gdf = gdf.dropna(subset=["block_id"]).copy()
    gdf = gdf[["block_id", "tract_id", "geometry"]].drop_duplicates("block_id")

    print(f"Loaded {len(gdf):,} census blocks")
    return gdf


async def fetch_neighborhoods(phl: Philly, refresh: bool) -> gpd.GeoDataFrame:
    raw_path = RAW_DIR / "philadelphia_neighborhoods.parquet"
    if raw_path.exists() and not refresh:
        print(f"Loading cached neighborhoods from {raw_path}")
        gdf = gpd.read_parquet(raw_path)
    else:
        print("Fetching Philadelphia Neighborhoods (GeoJSON)...")
        payload = await phl.load(
            "Philadelphia Neighborhoods",
            resource_name="Philadelphia Neighborhoods GeoJSON",
            use_cache=True,
        )
        assert isinstance(payload, dict) and "features" in payload
        gdf = gpd.GeoDataFrame.from_features(payload["features"], crs="EPSG:4326")
        gdf.to_parquet(raw_path, index=False)
        print(f"Saved raw neighborhoods to {raw_path}")

    gdf["neighborhood_name"] = clean_text(gdf.get("LISTNAME"))
    gdf["neighborhood_name"] = gdf["neighborhood_name"].fillna(
        clean_text(gdf.get("MAPNAME"))
    )
    gdf["neighborhood_name"] = gdf["neighborhood_name"].fillna(
        clean_text(gdf.get("NAME"))
    )
    gdf["neighborhood_name"] = gdf["neighborhood_name"].fillna("Unknown")
    gdf["neighborhood_id"] = (
        gdf["neighborhood_name"]
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "-", regex=True)
        .str.strip("-")
    )
    gdf = gdf[["neighborhood_id", "neighborhood_name", "geometry"]].drop_duplicates(
        "neighborhood_id"
    )

    print(f"Loaded {len(gdf):,} neighborhoods")
    return gdf


async def fetch_311(
    phl: Philly, start_date: date, end_exclusive: date, refresh: bool
) -> pd.DataFrame:
    raw_path = (
        RAW_DIR
        / f"requests_311_{start_date}_to_{end_exclusive - timedelta(days=1)}.parquet"
    )
    if raw_path.exists() and not refresh:
        print(f"Loading cached 311 data from {raw_path}")
        return pd.read_parquet(raw_path)

    print(f"Fetching 311 requests from {start_date} to {end_exclusive}...")
    where = (
        f"requested_datetime >= '{start_date.isoformat()}' "
        f"AND requested_datetime < '{end_exclusive.isoformat()}'"
    )
    columns = [
        "objectid",
        "service_request_id",
        "subject",
        "service_name",
        "status",
        "agency_responsible",
        "requested_datetime",
        "closed_datetime",
        "address",
        "lat",
        "lon",
    ]
    rows = await phl.load(
        "311 Service and Information Requests",
        where=where,
        columns=columns,
        use_cache=True,
    )
    df = pd.DataFrame(rows)
    df.to_parquet(raw_path, index=False)
    print(f"Saved raw 311 data to {raw_path} ({len(df):,} rows)")
    return df


async def fetch_crime(
    phl: Philly, start_date: date, end_exclusive: date, refresh: bool
) -> pd.DataFrame:
    raw_path = (
        RAW_DIR / f"crime_{start_date}_to_{end_exclusive - timedelta(days=1)}.parquet"
    )
    if raw_path.exists() and not refresh:
        print(f"Loading cached crime data from {raw_path}")
        return pd.read_parquet(raw_path)

    print(f"Fetching crime incidents from {start_date} to {end_exclusive}...")
    where = (
        f"dispatch_date_time >= '{start_date.isoformat()}' "
        f"AND dispatch_date_time < '{end_exclusive.isoformat()}'"
    )
    columns = [
        "objectid",
        "dc_key",
        "dispatch_date_time",
        "location_block",
        "text_general_code",
        "point_x",
        "point_y",
    ]
    rows = await phl.load(
        "Crime Incidents",
        where=where,
        columns=columns,
        use_cache=True,
    )
    df = pd.DataFrame(rows)
    df.to_parquet(raw_path, index=False)
    print(f"Saved raw crime data to {raw_path} ({len(df):,} rows)")
    return df


def spatially_assign_blocks(
    df: pd.DataFrame,
    blocks: gpd.GeoDataFrame,
    *,
    lat_col: str,
    lng_col: str,
    label: str,
    record_id_col: str,
) -> pd.DataFrame:
    working = df.copy()
    working[lat_col] = pd.to_numeric(working[lat_col], errors="coerce")
    working[lng_col] = pd.to_numeric(working[lng_col], errors="coerce")

    working = working.dropna(subset=[lat_col, lng_col]).copy()
    working = working[
        working[lat_col].between(39.7, 40.2) & working[lng_col].between(-75.4, -74.9)
    ].copy()
    working["__dedupe_id"] = clean_id(working[record_id_col]).fillna(
        pd.Series(working.index.astype(str), index=working.index, dtype="string")
    )

    print(f"Spatially joining {len(working):,} {label} points to census blocks...")

    points = gpd.GeoDataFrame(
        working,
        geometry=gpd.points_from_xy(working[lng_col], working[lat_col]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(
        points,
        blocks[["block_id", "tract_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    before_dedup = len(joined)
    joined = joined.drop_duplicates(subset=["__dedupe_id"]).copy()
    print(
        f"Matched {len(joined):,} {label} records to blocks "
        f"({before_dedup - len(joined):,} boundary duplicates dropped)"
    )

    return pd.DataFrame(
        joined.drop(columns=["geometry", "__dedupe_id"], errors="ignore")
    )


def normalize_311(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["service_request_id"] = clean_id(working["service_request_id"])
    working["objectid"] = clean_id(working["objectid"])
    working["source_record_id"] = working["service_request_id"].fillna(
        working["objectid"]
    )
    working["event_date"] = pd.to_datetime(
        working["requested_datetime"], errors="coerce", utc=True
    )
    working["event_end_date"] = pd.to_datetime(
        working["closed_datetime"], errors="coerce", utc=True
    )
    working["title"] = clean_text(working["service_name"]).fillna(
        clean_text(working["subject"])
    )
    working["title"] = working["title"].fillna("311 request")
    working["description"] = make_description(
        clean_text(working["status"]), clean_text(working["agency_responsible"])
    )

    normalized = pd.DataFrame(
        {
            "event_id": "311:" + working["source_record_id"].fillna("unknown"),
            "source_dataset": "311 Service and Information Requests",
            "source_record_id": working["source_record_id"],
            "event_type": "service_request",
            "event_subtype": clean_text(working["service_name"]),
            "event_date": working["event_date"],
            "event_end_date": working["event_end_date"],
            "title": working["title"],
            "description": working["description"],
            "address": clean_text(working["address"]),
            "lat": pd.to_numeric(working["lat"], errors="coerce"),
            "lng": pd.to_numeric(working["lon"], errors="coerce"),
            "block_id": clean_id(working["block_id"], width=15),
            "tract_id": clean_id(working["tract_id"], width=11),
            "status": clean_text(working["status"]),
            "severity": pd.Series(pd.NA, index=working.index, dtype="string"),
            "tags": clean_text(working["service_name"]),
        }
    )

    normalized = normalized.dropna(
        subset=["source_record_id", "event_date", "block_id"]
    )
    normalized = normalized.sort_values(
        ["event_date", "event_id"], ascending=[False, True]
    )
    return normalized.reset_index(drop=True)


def normalize_crime(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["dc_key"] = clean_id(working["dc_key"])
    working["objectid"] = clean_id(working["objectid"])
    working["source_record_id"] = working["dc_key"].fillna(working["objectid"])
    working["event_date"] = pd.to_datetime(
        working["dispatch_date_time"], errors="coerce", utc=True
    )
    working["title"] = clean_text(working["text_general_code"]).fillna("Crime incident")
    working["description"] = clean_text(working["location_block"])

    normalized = pd.DataFrame(
        {
            "event_id": "crime:" + working["source_record_id"].fillna("unknown"),
            "source_dataset": "Crime Incidents",
            "source_record_id": working["source_record_id"],
            "event_type": "crime_incident",
            "event_subtype": clean_text(working["text_general_code"]),
            "event_date": working["event_date"],
            "event_end_date": pd.Series(
                pd.NaT, index=working.index, dtype="datetime64[ns, UTC]"
            ),
            "title": working["title"],
            "description": working["description"],
            "address": clean_text(working["location_block"]),
            "lat": pd.to_numeric(working["point_y"], errors="coerce"),
            "lng": pd.to_numeric(working["point_x"], errors="coerce"),
            "block_id": clean_id(working["block_id"], width=15),
            "tract_id": clean_id(working["tract_id"], width=11),
            "status": pd.Series(pd.NA, index=working.index, dtype="string"),
            "severity": pd.Series(pd.NA, index=working.index, dtype="string"),
            "tags": clean_text(working["text_general_code"]),
        }
    )

    normalized = normalized.dropna(
        subset=["source_record_id", "event_date", "block_id"]
    )
    normalized = normalized.sort_values(
        ["event_date", "event_id"], ascending=[False, True]
    )
    return normalized.reset_index(drop=True)


def assign_blocks_to_neighborhoods(
    blocks: gpd.GeoDataFrame, neighborhoods: gpd.GeoDataFrame
) -> pd.DataFrame:
    print("Assigning census blocks to neighborhoods...")
    centroids = blocks[["block_id", "tract_id", "geometry"]].copy()
    centroids = centroids.to_crs("EPSG:3857")
    centroids["geometry"] = centroids.geometry.centroid
    centroids = centroids.to_crs("EPSG:4326")

    joined = gpd.sjoin(
        centroids,
        neighborhoods[["neighborhood_id", "neighborhood_name", "geometry"]],
        how="left",
        predicate="within",
    )
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    missing_mask = joined["neighborhood_id"].isna()
    if missing_mask.any():
        missing_blocks = joined.loc[
            missing_mask, ["block_id", "tract_id", "geometry"]
        ].copy()
        missing_blocks = missing_blocks.to_crs("EPSG:3857")
        neighborhoods_projected = neighborhoods[
            ["neighborhood_id", "neighborhood_name", "geometry"]
        ].to_crs("EPSG:3857")
        nearest = gpd.sjoin_nearest(
            missing_blocks,
            neighborhoods_projected,
            how="left",
            distance_col="distance",
        )
        nearest = nearest.drop(columns=["index_right", "distance"], errors="ignore")
        joined.loc[missing_mask, ["neighborhood_id", "neighborhood_name"]] = nearest[
            ["neighborhood_id", "neighborhood_name"]
        ].to_numpy()

    mapping = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))
    print(
        f"Assigned {mapping['neighborhood_id'].notna().sum():,} blocks to neighborhoods"
    )
    return mapping


def build_block_summary(
    events: pd.DataFrame, blocks: gpd.GeoDataFrame, as_of_date: date
) -> pd.DataFrame:
    summary = blocks[["block_id", "tract_id"]].drop_duplicates("block_id").copy()
    as_of_ts = pd.Timestamp(as_of_date).tz_localize(UTC)

    for days in [30, 90, 365]:
        cutoff = as_of_ts - pd.Timedelta(days=days)
        counts = (
            events.loc[events["event_date"] >= cutoff]
            .groupby("block_id")
            .size()
            .rename(f"events_{days}d")
            .reset_index()
        )
        summary = summary.merge(counts, on="block_id", how="left")

    cutoff_365 = as_of_ts - pd.Timedelta(days=365)
    for event_type, column_name in {
        "service_request": "service_requests_365d",
        "crime_incident": "crime_incidents_365d",
    }.items():
        counts = (
            events.loc[
                (events["event_date"] >= cutoff_365)
                & (events["event_type"] == event_type)
            ]
            .groupby("block_id")
            .size()
            .rename(column_name)
            .reset_index()
        )
        summary = summary.merge(counts, on="block_id", how="left")

    last_event = (
        events.groupby("block_id")["event_date"]
        .max()
        .rename("last_event_date")
        .reset_index()
    )
    summary = summary.merge(last_event, on="block_id", how="left")

    count_columns = [
        "events_30d",
        "events_90d",
        "events_365d",
        "service_requests_365d",
        "crime_incidents_365d",
    ]
    for column in count_columns:
        summary[column] = summary[column].fillna(0).astype("int32")

    return summary.sort_values(
        ["events_365d", "last_event_date", "block_id"], ascending=[False, False, True]
    ).reset_index(drop=True)


def derive_display_names(events: pd.DataFrame) -> pd.DataFrame:
    labeled = events.copy()
    labeled["address"] = clean_text(labeled["address"])
    labeled = labeled.dropna(subset=["block_id", "address", "event_date"])
    labeled = labeled.sort_values(["event_date", "event_id"], ascending=[False, True])
    labeled = labeled.drop_duplicates(subset=["block_id"])
    return labeled[["block_id", "address"]].rename(columns={"address": "display_name"})


def build_neighborhood_summary(
    block_summary: pd.DataFrame,
    blocks_with_neighborhoods: gpd.GeoDataFrame,
    neighborhoods: gpd.GeoDataFrame,
) -> pd.DataFrame:
    joined = blocks_with_neighborhoods.drop(
        columns=["geometry"], errors="ignore"
    ).merge(
        block_summary,
        on=["block_id", "tract_id"],
        how="left",
    )

    for column in [
        "events_30d",
        "events_90d",
        "events_365d",
        "service_requests_365d",
        "crime_incidents_365d",
    ]:
        joined[column] = joined[column].fillna(0).astype("int32")

    tract_lists = (
        joined.groupby("neighborhood_id")["tract_id"]
        .agg(lambda values: sorted({str(value) for value in values if pd.notna(value)}))
        .rename("tract_ids")
        .reset_index()
    )
    block_lists = (
        joined.groupby("neighborhood_id")["block_id"]
        .agg(lambda values: sorted({str(value) for value in values if pd.notna(value)}))
        .rename("block_ids")
        .reset_index()
    )

    summary = (
        joined.groupby(["neighborhood_id", "neighborhood_name"], dropna=False)
        .agg(
            block_count=("block_id", "nunique"),
            events_30d=("events_30d", "sum"),
            events_90d=("events_90d", "sum"),
            events_365d=("events_365d", "sum"),
            service_requests_365d=("service_requests_365d", "sum"),
            crime_incidents_365d=("crime_incidents_365d", "sum"),
            last_event_date=("last_event_date", "max"),
        )
        .reset_index()
    )
    summary = summary.merge(tract_lists, on="neighborhood_id", how="left")
    summary = summary.merge(block_lists, on="neighborhood_id", how="left")

    neighborhoods_summary = neighborhoods[
        ["neighborhood_id", "neighborhood_name"]
    ].merge(
        summary,
        on=["neighborhood_id", "neighborhood_name"],
        how="left",
    )

    for column in [
        "block_count",
        "events_30d",
        "events_90d",
        "events_365d",
        "service_requests_365d",
        "crime_incidents_365d",
    ]:
        neighborhoods_summary[column] = (
            neighborhoods_summary[column].fillna(0).astype("int32")
        )

    neighborhoods_summary["tract_ids"] = neighborhoods_summary["tract_ids"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    neighborhoods_summary["block_ids"] = neighborhoods_summary["block_ids"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    return neighborhoods_summary.sort_values(
        ["events_365d", "neighborhood_name"], ascending=[False, True]
    ).reset_index(drop=True)


def export_app_data(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    blocks: gpd.GeoDataFrame,
    neighborhoods: gpd.GeoDataFrame,
    neighborhood_summary: pd.DataFrame,
) -> None:
    print("Exporting frontend data to app/data/...")

    tracts_dir = APP_DATA_DIR / "tracts"
    tracts_dir.mkdir(parents=True, exist_ok=True)
    for existing_file in tracts_dir.glob("*.json"):
        existing_file.unlink()

    blocks_export = blocks.merge(summary, on=["block_id", "tract_id"], how="left")
    for column in [
        "events_30d",
        "events_90d",
        "events_365d",
        "service_requests_365d",
        "crime_incidents_365d",
    ]:
        blocks_export[column] = blocks_export[column].fillna(0).astype("int32")
    blocks_export["display_name"] = blocks_export["display_name"].fillna("")
    blocks_export["last_event_date"] = blocks_export["last_event_date"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    blocks_export["geometry"] = blocks_export.geometry.simplify(
        0.00001, preserve_topology=True
    )

    blocks_geojson_path = APP_DATA_DIR / "blocks.geojson"
    blocks_geojson_path.write_text(
        blocks_export.to_json(drop_id=True), encoding="utf-8"
    )

    neighborhoods_export = neighborhoods.merge(
        neighborhood_summary,
        on=["neighborhood_id", "neighborhood_name"],
        how="left",
    )
    for column in [
        "block_count",
        "events_30d",
        "events_90d",
        "events_365d",
        "service_requests_365d",
        "crime_incidents_365d",
    ]:
        neighborhoods_export[column] = (
            neighborhoods_export[column].fillna(0).astype("int32")
        )
    neighborhoods_export["last_event_date"] = neighborhoods_export[
        "last_event_date"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    neighborhoods_export["geometry"] = neighborhoods_export.geometry.simplify(
        0.00005, preserve_topology=True
    )

    neighborhoods_geojson_path = APP_DATA_DIR / "neighborhoods.geojson"
    neighborhoods_geojson_path.write_text(
        neighborhoods_export.to_json(drop_id=True), encoding="utf-8"
    )

    neighborhood_index_path = APP_DATA_DIR / "neighborhood_index.json"
    neighborhood_index_payload = {
        row["neighborhood_id"]: {
            "name": row["neighborhood_name"],
            "tract_ids": row["tract_ids"],
            "block_ids": row["block_ids"],
        }
        for _, row in neighborhood_summary.iterrows()
    }
    neighborhood_index_path.write_text(
        json.dumps(neighborhood_index_payload, separators=(",", ":")),
        encoding="utf-8",
    )

    address_lookup = (
        events[["address", "block_id", "event_date"]]
        .dropna(subset=["address", "block_id", "event_date"])
        .copy()
    )
    address_lookup["address"] = clean_text(address_lookup["address"]).str.upper()
    address_lookup = address_lookup.dropna(subset=["address"])
    address_lookup = address_lookup.sort_values("event_date", ascending=False)
    address_lookup = address_lookup.drop_duplicates(subset=["address"])
    address_lookup_path = APP_DATA_DIR / "address_index.json"
    address_lookup_path.write_text(
        json.dumps(
            dict(
                zip(address_lookup["address"], address_lookup["block_id"], strict=False)
            ),
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    events_export = events[
        [
            "block_id",
            "tract_id",
            "event_type",
            "event_date",
            "title",
            "description",
            "source_record_id",
        ]
    ].copy()
    events_export["event_date"] = events_export["event_date"].dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    events_export = events_export.rename(
        columns={
            "block_id": "b",
            "tract_id": "tr",
            "event_type": "t",
            "event_date": "d",
            "title": "ti",
            "description": "de",
            "source_record_id": "r",
        }
    )

    tract_counts: dict[str, int] = {}
    for tract_id, tract_events in events_export.groupby("tr", dropna=True):
        tract_id = str(tract_id)
        tract_path = tracts_dir / f"{tract_id}.json"
        tract_path.write_text(tract_events.to_json(orient="records"), encoding="utf-8")
        tract_counts[tract_id] = int(len(tract_events))

    latest_event = events["event_date"].max()
    app_manifest = {
        "blocks_geojson": blocks_geojson_path.name,
        "neighborhoods_geojson": neighborhoods_geojson_path.name,
        "neighborhood_index": neighborhood_index_path.name,
        "address_index": address_lookup_path.name,
        "tract_count": len(tract_counts),
        "blocks_with_events": int(
            summary.loc[summary["events_365d"] > 0, "block_id"].nunique()
        ),
        "neighborhoods_with_events": int(
            neighborhood_summary.loc[
                neighborhood_summary["events_365d"] > 0, "neighborhood_id"
            ].nunique()
        ),
        "total_events": int(len(events_export)),
        "event_types": sorted(events_export["t"].dropna().unique().tolist()),
        "as_of_date": latest_event.strftime("%Y-%m-%dT%H:%M:%SZ")
        if pd.notna(latest_event)
        else None,
    }
    (APP_DATA_DIR / "manifest.json").write_text(
        json.dumps(app_manifest, indent=2), encoding="utf-8"
    )


async def main() -> None:
    args = parse_args()
    ensure_dirs()

    start_date = parse_date_arg(args.start_date)
    end_date = parse_date_arg(args.end_date)
    if end_date < start_date:
        raise ValueError("end-date must be on or after start-date")
    end_exclusive = end_date + timedelta(days=1)

    phl = Philly()
    refresh = not args.skip_fetch

    blocks = await fetch_census_blocks(phl, refresh=refresh)
    neighborhoods = await fetch_neighborhoods(phl, refresh=refresh)
    block_neighborhoods = assign_blocks_to_neighborhoods(blocks, neighborhoods)
    blocks = blocks.merge(block_neighborhoods, on=["block_id", "tract_id"], how="left")
    requests_311 = await fetch_311(phl, start_date, end_exclusive, refresh=refresh)
    crime = await fetch_crime(phl, start_date, end_exclusive, refresh=refresh)
    requests_311_raw_count = len(requests_311)
    crime_raw_count = len(crime)

    requests_311_joined = spatially_assign_blocks(
        requests_311,
        blocks,
        lat_col="lat",
        lng_col="lon",
        label="311",
        record_id_col="service_request_id",
    )
    events_311 = normalize_311(requests_311_joined)
    silver_311_path = SILVER_DIR / "events_311.parquet"
    events_311.to_parquet(silver_311_path, index=False)
    print(f"Saved {len(events_311):,} normalized 311 events to {silver_311_path}")
    del requests_311, requests_311_joined
    gc.collect()

    crime_joined = spatially_assign_blocks(
        crime,
        blocks,
        lat_col="point_y",
        lng_col="point_x",
        label="crime",
        record_id_col="dc_key",
    )
    events_crime = normalize_crime(crime_joined)
    silver_crime_path = SILVER_DIR / "events_crime.parquet"
    events_crime.to_parquet(silver_crime_path, index=False)
    print(f"Saved {len(events_crime):,} normalized crime events to {silver_crime_path}")
    del crime, crime_joined
    gc.collect()

    block_log = pd.concat([events_311, events_crime], ignore_index=True)
    block_log = block_log.sort_values(
        ["event_date", "event_id"], ascending=[False, True]
    )
    block_log = block_log.reset_index(drop=True)

    block_log_path = GOLD_DIR / "block_log.parquet"
    block_log.to_parquet(block_log_path, index=False)
    print(f"Saved {len(block_log):,} events to {block_log_path}")

    block_summary = build_block_summary(block_log, blocks, as_of_date=end_date)
    block_summary = block_summary.merge(
        derive_display_names(block_log), on="block_id", how="left"
    )
    block_summary_path = GOLD_DIR / "block_summary.parquet"
    block_summary.to_parquet(block_summary_path, index=False)
    print(
        f"Saved block summary for {len(block_summary):,} blocks to {block_summary_path}"
    )

    neighborhood_summary = build_neighborhood_summary(
        block_summary, blocks, neighborhoods
    )
    neighborhood_summary_path = GOLD_DIR / "neighborhood_summary.parquet"
    neighborhood_summary.to_parquet(neighborhood_summary_path, index=False)
    print(
        "Saved neighborhood summary for "
        f"{len(neighborhood_summary):,} neighborhoods to {neighborhood_summary_path}"
    )

    blocks_path = GOLD_DIR / "census_blocks.parquet"
    blocks.to_parquet(blocks_path, index=False)
    print(f"Saved block geometry to {blocks_path}")

    neighborhoods_path = GOLD_DIR / "neighborhoods.parquet"
    neighborhoods.to_parquet(neighborhoods_path, index=False)
    print(f"Saved neighborhood geometry to {neighborhoods_path}")

    manifest = {
        "built_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "datasets": {
            "census_blocks": int(len(blocks)),
            "neighborhoods": int(len(neighborhoods)),
            "requests_311_raw": int(requests_311_raw_count),
            "requests_311_matched": int(len(events_311)),
            "crime_raw": int(crime_raw_count),
            "crime_matched": int(len(events_crime)),
        },
        "gold": {
            "total_events": int(len(block_log)),
            "blocks_with_events": int(block_log["block_id"].nunique()),
            "neighborhoods_with_events": int(
                neighborhood_summary.loc[
                    neighborhood_summary["events_365d"] > 0, "neighborhood_id"
                ].nunique()
            ),
        },
    }
    manifest_path = GOLD_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved build manifest to {manifest_path}")

    if args.export_app_data:
        export_app_data(
            block_summary, block_log, blocks, neighborhoods, neighborhood_summary
        )

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
