"""Fetch Philadelphia data and compute block-level grades for report card visualization."""

import asyncio
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
from scipy.stats import percentileofscore
from shapely import wkt

from philly import Philly


def score_to_grade(score: float) -> str:
    """Convert a percentile score (0-100, higher is better) to a letter grade."""
    if score >= 80:
        return "A"
    elif score >= 60:
        return "B"
    elif score >= 40:
        return "C"
    elif score >= 20:
        return "D"
    else:
        return "F"


async def fetch_census_blocks() -> gpd.GeoDataFrame:
    """Fetch Census Blocks geometry."""
    phl = Philly()

    print("Fetching Census Blocks...")
    # Use 2010 census blocks which have GEOID10
    blocks = await phl.load(
        "Census Blocks", resource_name="Census Blocks - 2010 (GeoJSON)"
    )

    # Type assertion - blocks is a GeoJSON FeatureCollection dict
    assert isinstance(blocks, dict) and "features" in blocks
    gdf = gpd.GeoDataFrame.from_features(blocks["features"], crs="EPSG:4326")

    # Calculate area in sq km (project to PA State Plane for accurate area)
    gdf_proj = gdf.to_crs("EPSG:2272")  # PA State Plane South (feet)
    gdf["area_sqm"] = gdf_proj.geometry.area * 0.0929  # sq feet to sq meters
    gdf["area_sqkm"] = gdf["area_sqm"] / 1_000_000

    # Get land area from census data (ALAND10 is in sq meters)
    if "ALAND10" in gdf.columns:
        gdf["land_area"] = pd.to_numeric(gdf["ALAND10"], errors="coerce").fillna(0)
    else:
        gdf["land_area"] = gdf["area_sqm"]  # Fallback to computed area

    # Flag residential blocks (has land, not just water)
    # A block is considered "residential" if it has > 100 sq meters of land
    gdf["is_residential"] = gdf["land_area"] > 100

    # Use GEOID10 as the block identifier
    if "GEOID10" in gdf.columns:
        gdf["block_id"] = gdf["GEOID10"]
    elif "GEOID" in gdf.columns:
        gdf["block_id"] = gdf["GEOID"]
    else:
        # Try to find any ID column
        id_cols = [c for c in gdf.columns if "ID" in c.upper() or "GEOID" in c.upper()]
        if id_cols:
            gdf["block_id"] = gdf[id_cols[0]]
        else:
            gdf["block_id"] = gdf.index.astype(str)

    residential_count = gdf["is_residential"].sum()
    print(f"  Got {len(gdf):,} census blocks ({residential_count:,} residential)")
    return gdf


async def fetch_litter_index() -> pd.DataFrame:
    """Fetch Litter Index as geospatial data for spatial joining."""
    phl = Philly()

    print("Fetching Litter Index...")
    # Litter index uses street segments (SEG_ID), not census blocks
    # We'll load the GeoJSON version to do spatial joins
    litter = await phl.load(
        "Litter Index",
        resource_name="Litter Index Blocks Scores 2017 - 2018 (GeoJSON)",
    )

    # Type assertion
    assert isinstance(litter, dict) and "features" in litter
    df = gpd.GeoDataFrame.from_features(litter["features"], crs="EPSG:4326")
    print(f"  Got {len(df):,} litter index records")

    # Find score column
    score_cols = [c for c in df.columns if "score" in c.lower()]
    if score_cols:
        df["litter_score"] = pd.to_numeric(df[score_cols[0]], errors="coerce")
    else:
        df["litter_score"] = None

    return df


async def fetch_crime_incidents() -> pd.DataFrame:
    """Fetch 2024 crime incidents."""
    phl = Philly()

    print("Fetching Crime Incidents (2024)...")
    crime = await phl.load(
        "Crime Incidents",
        where="dispatch_date_time >= '2024-01-01' AND dispatch_date_time < '2025-01-01'",
    )

    df = pd.DataFrame(crime)
    print(f"  Got {len(df):,} crime records")

    # Get coordinates
    if "point_x" in df.columns and "point_y" in df.columns:
        df["lng"] = pd.to_numeric(df["point_x"], errors="coerce")
        df["lat"] = pd.to_numeric(df["point_y"], errors="coerce")
    elif "lng" in df.columns and "lat" in df.columns:
        df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    elif "lon" in df.columns and "lat" in df.columns:
        df["lng"] = pd.to_numeric(df["lon"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    # Filter valid coordinates
    df = df.dropna(subset=["lat", "lng"])
    df = df[(df["lat"] != 0) & (df["lng"] != 0)]

    print(f"  {len(df):,} records with valid coordinates")
    return df


async def fetch_311_requests() -> pd.DataFrame:
    """Fetch 2024 311 requests with response times."""
    phl = Philly()

    print("Fetching 311 Requests (2024)...")
    requests = await phl.load(
        "311 Service and Information Requests",
        where="requested_datetime >= '2024-01-01' AND requested_datetime < '2025-01-01'",
    )

    df = pd.DataFrame(requests)
    print(f"  Got {len(df):,} 311 records")

    # Get coordinates
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lng"] = pd.to_numeric(df["lon"], errors="coerce")

    # Calculate response time in hours
    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")
    if "closed_datetime" in df.columns:
        df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], errors="coerce")
        df["response_hours"] = (
            df["closed_datetime"] - df["requested_datetime"]
        ).dt.total_seconds() / 3600
        # Filter out negative or extremely long response times
        df.loc[df["response_hours"] < 0, "response_hours"] = None
        df.loc[df["response_hours"] > 8760, "response_hours"] = None  # > 1 year

    # Filter valid coordinates
    df = df.dropna(subset=["lat", "lng"])
    df = df[(df["lat"] != 0) & (df["lng"] != 0)]

    print(f"  {len(df):,} records with valid coordinates")
    return df


async def fetch_parking_violations() -> pd.DataFrame:
    """Fetch parking violations (uses 2017 data - most recent available)."""
    phl = Philly()

    # Parking data on Carto only goes through 2017-12-31
    print("Fetching Parking Violations (2017)...")
    parking = await phl.load(
        "Parking Violations",
        resource_name="Parking Violations - 2018-present (CSV)",
        where="issue_datetime >= '2017-01-01' AND issue_datetime < '2018-01-01'",
    )

    df = pd.DataFrame(parking)
    print(f"  Got {len(df):,} parking records")

    if len(df) == 0:
        print("  Warning: No parking violation data available")
        return pd.DataFrame(columns=["lat", "lng"])

    # Get coordinates
    if "lat" in df.columns and "lon" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lng"] = pd.to_numeric(df["lon"], errors="coerce")
    else:
        print("  Warning: No coordinate columns found in parking data")
        return pd.DataFrame(columns=["lat", "lng"])

    # Filter valid coordinates
    df = df.dropna(subset=["lat", "lng"])
    df = df[(df["lat"] != 0) & (df["lng"] != 0)]

    print(f"  {len(df):,} records with valid coordinates")
    return df


def spatial_join_points_to_blocks(
    points_df: pd.DataFrame,
    blocks_gdf: gpd.GeoDataFrame,
    lat_col: str = "lat",
    lng_col: str = "lng",
) -> pd.DataFrame:
    """Join points to census blocks using spatial join."""
    print(f"  Spatial joining {len(points_df):,} points to blocks...")

    # Handle empty dataframe
    if len(points_df) == 0:
        print("  No points to join")
        return pd.DataFrame(columns=["block_id"])

    # Create GeoDataFrame from points
    points_gdf = gpd.GeoDataFrame(
        points_df,
        geometry=gpd.points_from_xy(points_df[lng_col], points_df[lat_col]),
        crs="EPSG:4326",
    )

    # Spatial join
    joined = gpd.sjoin(
        points_gdf, blocks_gdf[["block_id", "geometry"]], predicate="within"
    )

    print(f"  {len(joined):,} points matched to blocks")
    return joined


def calculate_block_grades(
    blocks_gdf: gpd.GeoDataFrame,
    crime_by_block: pd.DataFrame,
    service_by_block: pd.DataFrame,
    parking_by_block: pd.DataFrame,
    litter_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate grades for each block based on metrics."""

    print("\nCalculating block grades...")

    # Start with all blocks
    grades = blocks_gdf[["block_id", "area_sqkm", "is_residential"]].copy()

    # 1. Safety: crimes per sq km (lower is better)
    crime_counts = (
        crime_by_block.groupby("block_id").size().reset_index(name="crime_count")
    )
    grades = grades.merge(crime_counts, on="block_id", how="left")
    grades["crime_count"] = grades["crime_count"].fillna(0)
    grades["crime_density"] = grades["crime_count"] / grades["area_sqkm"].clip(
        lower=0.001
    )

    # Score: higher score = safer (inverted percentile)
    # Round to 1 decimal BEFORE grading to ensure display matches grade
    grades["safety_score"] = grades["crime_density"].apply(
        lambda x: round(
            100 - percentileofscore(grades["crime_density"].dropna(), x, kind="rank"),
            1,
        )
    )
    grades["safety_grade"] = grades["safety_score"].apply(score_to_grade)

    # 2. Responsiveness: median hours to close 311 requests (lower is better)
    service_response = (
        service_by_block.groupby("block_id")["response_hours"]
        .median()
        .reset_index(name="median_response_hours")
    )
    grades = grades.merge(service_response, on="block_id", how="left")

    # Score: higher score = more responsive (inverted percentile)
    # Round to 1 decimal BEFORE grading to ensure display matches grade
    valid_response = grades["median_response_hours"].dropna()
    grades["responsiveness_score"] = grades["median_response_hours"].apply(
        lambda x: round(100 - percentileofscore(valid_response, x, kind="rank"), 1)
        if pd.notna(x)
        else None
    )
    grades["responsiveness_grade"] = grades["responsiveness_score"].apply(
        lambda x: score_to_grade(x) if pd.notna(x) else "N/A"
    )

    # 3. Cleanliness: litter index score (1-4, lower is better)
    # litter_df may have block_id from spatial join, or may be empty
    if "block_id" in litter_df.columns and "litter_score" in litter_df.columns:
        # Average litter scores per block (multiple segments may fall in one block)
        litter_by_block = (
            litter_df.groupby("block_id")["litter_score"].mean().reset_index()
        )
        grades = grades.merge(litter_by_block, on="block_id", how="left")
    else:
        grades["litter_score"] = None

    # Score: higher score = cleaner (inverted, assuming 1=clean, 4=dirty)
    # Round to 1 decimal BEFORE grading to ensure display matches grade
    valid_litter = grades["litter_score"].dropna()
    if len(valid_litter) > 0:
        grades["cleanliness_score"] = grades["litter_score"].apply(
            lambda x: round(100 - percentileofscore(valid_litter, x, kind="rank"), 1)
            if pd.notna(x)
            else None
        )
    else:
        grades["cleanliness_score"] = None
    grades["cleanliness_grade"] = grades["cleanliness_score"].apply(
        lambda x: score_to_grade(x) if pd.notna(x) else "N/A"
    )

    # 4. Parking Karma: violations per sq km (lower is better)
    parking_counts = (
        parking_by_block.groupby("block_id").size().reset_index(name="parking_count")
    )
    grades = grades.merge(parking_counts, on="block_id", how="left")
    grades["parking_count"] = grades["parking_count"].fillna(0)
    grades["parking_density"] = grades["parking_count"] / grades["area_sqkm"].clip(
        lower=0.001
    )

    # Score: higher score = fewer violations
    # Round to 1 decimal BEFORE grading to ensure display matches grade
    grades["parking_score"] = grades["parking_density"].apply(
        lambda x: round(
            100 - percentileofscore(grades["parking_density"].dropna(), x, kind="rank"),
            1,
        )
    )
    grades["parking_grade"] = grades["parking_score"].apply(score_to_grade)

    # Calculate overall GPA (A=4, B=3, C=2, D=1, F=0)
    grade_to_gpa = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}

    def calc_gpa(row):
        grades_list = []
        for col in [
            "safety_grade",
            "responsiveness_grade",
            "cleanliness_grade",
            "parking_grade",
        ]:
            if row[col] != "N/A":
                grades_list.append(grade_to_gpa.get(row[col], 0))
        return sum(grades_list) / len(grades_list) if grades_list else None

    grades["gpa"] = grades.apply(calc_gpa, axis=1)

    # Overall grade based on GPA
    def gpa_to_grade(gpa):
        if pd.isna(gpa):
            return "N/A"
        if gpa >= 3.5:
            return "A"
        elif gpa >= 2.5:
            return "B"
        elif gpa >= 1.5:
            return "C"
        elif gpa >= 0.5:
            return "D"
        else:
            return "F"

    grades["overall_grade"] = grades["gpa"].apply(gpa_to_grade)

    print(f"  Calculated grades for {len(grades):,} blocks")

    # Print grade distribution
    print("\n  Grade distribution (overall):")
    for grade in ["A", "B", "C", "D", "F", "N/A"]:
        count = (grades["overall_grade"] == grade).sum()
        print(f"    {grade}: {count:,} blocks")

    return grades


def simplify_geometry(
    gdf: gpd.GeoDataFrame, tolerance: float = 0.0001
) -> gpd.GeoDataFrame:
    """Simplify geometries to reduce file size."""
    print(f"\nSimplifying geometries (tolerance={tolerance})...")
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].simplify(tolerance, preserve_topology=True)

    # Round coordinates to 5 decimals
    def round_coords(geom):
        if geom is None:
            return None
        return wkt.loads(wkt.dumps(geom, rounding_precision=5))

    gdf["geometry"] = gdf["geometry"].apply(round_coords)

    return gdf


async def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Fetch all data
    blocks_gdf = await fetch_census_blocks()
    litter_df = await fetch_litter_index()
    crime_df = await fetch_crime_incidents()
    service_df = await fetch_311_requests()
    parking_df = await fetch_parking_violations()

    # Spatial join points to blocks
    print("\nSpatial joining datasets to blocks...")
    crime_by_block = spatial_join_points_to_blocks(crime_df, blocks_gdf)
    service_by_block = spatial_join_points_to_blocks(service_df, blocks_gdf)
    parking_by_block = spatial_join_points_to_blocks(parking_df, blocks_gdf)

    # Spatial join litter index (line segments) to blocks
    print("  Spatial joining litter index to blocks...")
    if isinstance(litter_df, gpd.GeoDataFrame) and len(litter_df) > 0:
        # Join litter segments that intersect with blocks
        litter_by_block = gpd.sjoin(
            litter_df[["geometry", "litter_score"]],
            blocks_gdf[["block_id", "geometry"]],
            predicate="intersects",
        )
        print(f"  {len(litter_by_block):,} litter segments matched to blocks")
    else:
        litter_by_block = pd.DataFrame(columns=["block_id", "litter_score"])
        print("  No litter data to join")

    # Calculate grades
    grades_df = calculate_block_grades(
        blocks_gdf,
        crime_by_block,
        service_by_block,
        parking_by_block,
        litter_by_block,
    )

    # Export grades as JSON
    print("\nExporting grades.json...")
    grades_export = grades_df[
        [
            "block_id",
            "is_residential",
            "safety_score",
            "safety_grade",
            "responsiveness_score",
            "responsiveness_grade",
            "cleanliness_score",
            "cleanliness_grade",
            "parking_score",
            "parking_grade",
            "gpa",
            "overall_grade",
        ]
    ].copy()

    # Convert to dict format for compact JSON
    grades_dict = {}
    for _, row in grades_export.iterrows():
        entry = {
            "s": round(row["safety_score"], 1)
            if pd.notna(row["safety_score"])
            else None,
            "sg": row["safety_grade"],
            "r": round(row["responsiveness_score"], 1)
            if pd.notna(row["responsiveness_score"])
            else None,
            "rg": row["responsiveness_grade"],
            "c": round(row["cleanliness_score"], 1)
            if pd.notna(row["cleanliness_score"])
            else None,
            "cg": row["cleanliness_grade"],
            "p": round(row["parking_score"], 1)
            if pd.notna(row["parking_score"])
            else None,
            "pg": row["parking_grade"],
            "gpa": round(row["gpa"], 2) if pd.notna(row["gpa"]) else None,
            "g": row["overall_grade"],
        }
        # Only add residential flag if False (saves space, True is default)
        if not row["is_residential"]:
            entry["nr"] = 1  # non-residential flag
        grades_dict[row["block_id"]] = entry

    grades_path = output_dir / "grades.json"
    with open(grades_path, "w") as f:
        json.dump(grades_dict, f, separators=(",", ":"))
    print(f"  Saved to {grades_path} ({grades_path.stat().st_size / 1024:.1f} KB)")

    # Export simplified GeoJSON
    print("\nExporting blocks.geojson...")
    blocks_export = blocks_gdf[["block_id", "geometry"]].copy()
    blocks_export = simplify_geometry(blocks_export)

    # Merge overall grade for choropleth coloring
    blocks_export = blocks_export.merge(
        grades_df[["block_id", "overall_grade"]], on="block_id", how="left"
    )

    blocks_path = output_dir / "blocks.geojson"
    blocks_export.to_file(blocks_path, driver="GeoJSON")
    print(
        f"  Saved to {blocks_path} ({blocks_path.stat().st_size / 1024 / 1024:.2f} MB)"
    )

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
