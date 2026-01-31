"""Process Indego trip data for animated flow visualization."""

import pandas as pd
from pathlib import Path
import json

def main():
    data_dir = Path(__file__).parent / "data"

    # Load trip data
    print("Loading trip data...")
    df = pd.read_csv(data_dir / "indego-trips-2024-q4.csv")
    print(f"Loaded {len(df):,} trips")

    # Parse timestamps
    df["start_time"] = pd.to_datetime(df["start_time"], format="%m/%d/%Y %H:%M")
    df["hour"] = df["start_time"].dt.hour
    df["day_of_week"] = df["start_time"].dt.dayofweek  # 0=Monday
    df["is_weekend"] = df["day_of_week"] >= 5

    # Filter to valid trips (different start/end stations)
    one_way = df[df["trip_route_category"] == "One Way"].copy()
    print(f"One-way trips: {len(one_way):,}")

    # Create OD pair key
    one_way["od_pair"] = one_way["start_station"].astype(str) + "-" + one_way["end_station"].astype(str)

    # Aggregate by OD pair and hour (weekday only for clearer patterns)
    weekday = one_way[~one_way["is_weekend"]]
    print(f"Weekday one-way trips: {len(weekday):,}")

    # Group by OD pair and hour
    flows = (
        weekday.groupby(["od_pair", "hour", "start_station", "end_station",
                         "start_lat", "start_lon", "end_lat", "end_lon"])
        .agg(
            count=("trip_id", "size"),
            avg_duration=("duration", "mean")
        )
        .reset_index()
    )

    print(f"Unique OD-hour combinations: {len(flows):,}")

    # Keep only flows with at least 3 trips (reduce noise)
    flows = flows[flows["count"] >= 3]
    print(f"After filtering (>=3 trips): {len(flows):,}")

    # Prepare for JSON export
    flow_data = []
    for _, row in flows.iterrows():
        flow_data.append({
            "hour": int(row["hour"]),
            "from": [float(row["start_lon"]), float(row["start_lat"])],
            "to": [float(row["end_lon"]), float(row["end_lat"])],
            "count": int(row["count"]),
            "duration": round(row["avg_duration"], 1)
        })

    # Save to JSON
    output_path = data_dir / "bike_flows.json"
    with open(output_path, "w") as f:
        json.dump(flow_data, f)

    print(f"\nSaved {len(flow_data):,} flows to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Show some stats
    print("\nFlows per hour:")
    hour_counts = flows.groupby("hour")["count"].sum().sort_index()
    for h, c in hour_counts.items():
        bar = "█" * (c // 500)
        print(f"  {h:02d}:00  {c:5,} {bar}")


if __name__ == "__main__":
    main()
