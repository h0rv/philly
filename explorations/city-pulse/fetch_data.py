"""Fetch Philadelphia activity data and export to parquet for visualization."""

import asyncio
from pathlib import Path

import pandas as pd

from philly import Philly


async def fetch_crime_hourly() -> pd.DataFrame:
    """Fetch crime incidents and aggregate by hour and location."""
    phl = Philly()

    print("Fetching crime incidents (2024)...")
    # Get 2024 crime data with coordinates
    crime = await phl.load(
        "Crime Incidents",
        where="dispatch_date >= '2024-01-01'",
    )

    df = pd.DataFrame(crime)

    # Parse datetime and extract hour - hour may already be a column
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    else:
        df["dispatch_date_time"] = pd.to_datetime(df["dispatch_date_time"], errors="coerce")
        df["hour"] = df["dispatch_date_time"].dt.hour

    df["dispatch_date_time"] = pd.to_datetime(df["dispatch_date_time"], errors="coerce")
    df["day_of_week"] = df["dispatch_date_time"].dt.dayofweek  # 0=Monday

    # Handle different coordinate column names
    if "point_x" in df.columns and "point_y" in df.columns:
        df["lng"] = pd.to_numeric(df["point_x"], errors="coerce")
        df["lat"] = pd.to_numeric(df["point_y"], errors="coerce")
    elif "lon" in df.columns:
        df["lng"] = df["lon"]

    # Keep only rows with valid coordinates
    df = df.dropna(subset=["lat", "lng", "hour"])
    df = df[df["lat"] != 0]
    df = df[df["lng"] != 0]

    # Select columns we need
    result = df[["lat", "lng", "hour", "day_of_week", "text_general_code"]].copy()
    result["type"] = "crime"

    print(f"  Got {len(result):,} crime records")
    return result


async def fetch_parking_hourly() -> pd.DataFrame:
    """Fetch parking violations and aggregate by hour and location."""
    phl = Philly()

    print("Fetching parking violations (2024)...")
    # Use the 2018-present resource with server-side filtering
    parking = await phl.load(
        "Parking Violations",
        resource="Parking Violations - 2018-present (CSV)",
        where="issue_datetime >= '2024-01-01'",
    )

    df = pd.DataFrame(parking)
    print(f"  Loaded {len(df):,} parking records")

    if df.empty:
        print("  Warning: No parking data returned")
        return pd.DataFrame(columns=["lat", "lng", "hour", "day_of_week", "text_general_code", "type"])

    # Parse datetime and extract hour
    df["issue_datetime"] = pd.to_datetime(df["issue_datetime"], errors="coerce")
    df["hour"] = df["issue_datetime"].dt.hour
    df["day_of_week"] = df["issue_datetime"].dt.dayofweek

    # Rename lon to lng for consistency and convert to numeric
    if "lon" in df.columns:
        df["lng"] = pd.to_numeric(df["lon"], errors="coerce")
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    # Keep only rows with valid coordinates
    df = df.dropna(subset=["lat", "lng", "hour"])
    df = df[df["lat"] != 0]
    df = df[df["lng"] != 0]

    # Select columns we need
    result = df[["lat", "lng", "hour", "day_of_week"]].copy()
    result["text_general_code"] = "PARKING"
    result["type"] = "parking"

    print(f"  Got {len(result):,} parking records with coordinates")
    return result


async def fetch_311_hourly() -> pd.DataFrame:
    """Fetch 311 requests by hour and location."""
    phl = Philly()

    print("Fetching 311 requests (2024)...")
    # 311 data supports server-side filtering via Carto
    requests = await phl.load(
        "311 Service and Information Requests",
        where="requested_datetime >= '2024-01-01'",
    )

    df = pd.DataFrame(requests)
    print(f"  Loaded {len(df):,} 311 records")

    # Parse datetime and extract hour
    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")
    df["hour"] = df["requested_datetime"].dt.hour
    df["day_of_week"] = df["requested_datetime"].dt.dayofweek

    # Convert coordinates to numeric
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # Keep only rows with valid coordinates
    df = df.dropna(subset=["lat", "lon", "hour"])
    df = df[df["lat"] != 0]
    df = df[df["lon"] != 0]

    # Rename lon to lng for consistency
    df = df.rename(columns={"lon": "lng"})

    # Select columns we need
    result = df[["lat", "lng", "hour", "day_of_week", "service_name"]].copy()
    result = result.rename(columns={"service_name": "text_general_code"})
    result["type"] = "311"

    print(f"  Got {len(result):,} 311 records with coordinates")
    return result


async def main():
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    # Fetch all data
    crime_df = await fetch_crime_hourly()
    parking_df = await fetch_parking_hourly()
    requests_df = await fetch_311_hourly()

    # Combine all data
    print("\nCombining datasets...")
    combined = pd.concat([crime_df, parking_df, requests_df], ignore_index=True)
    print(f"Total records: {len(combined):,}")

    # Create hourly aggregates for heatmap (grid-based)
    print("\nCreating hourly aggregates...")
    # Round coordinates to create grid cells (~100m resolution)
    combined["lat_grid"] = (combined["lat"] * 100).round() / 100
    combined["lng_grid"] = (combined["lng"] * 100).round() / 100

    # Aggregate by grid cell, hour, and type
    hourly = (
        combined.groupby(["lat_grid", "lng_grid", "hour", "type"])
        .size()
        .reset_index(name="count")
    )
    hourly = hourly.rename(columns={"lat_grid": "lat", "lng_grid": "lng"})

    print(f"Aggregated to {len(hourly):,} grid cells")

    # Save to parquet
    output_path = output_dir / "philly_activity.parquet"
    hourly.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    # Also save raw points for detailed view (sample for file size)
    print("\nSaving sample of raw points...")
    sample = combined.sample(n=min(100_000, len(combined)), random_state=42)
    sample_path = output_dir / "philly_points_sample.parquet"
    sample.to_parquet(sample_path, index=False)
    print(f"Saved {len(sample):,} sample points to {sample_path}")


if __name__ == "__main__":
    asyncio.run(main())
