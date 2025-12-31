# %% [markdown]
# # The 311 Gap: Philadelphia's Service Inequality - With Demographics
#
# ## Enhanced Investigation with Census Income Data
#
# This enhanced version adds census demographic data to test the correlation
# between neighborhood income and city service response times.
#
# **New Features:**
# - Loads census income data from American Community Survey
# - Calculates violations per capita
# - Tests statistical correlation between income and service quality
# - Identifies income-based disparities

# %% Import libraries
import asyncio
import sys
import warnings
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Add Philly library to path and import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from philly import Philly  # noqa: E402

warnings.filterwarnings("ignore")

# Set up plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

print("Libraries loaded successfully (including scipy for statistical tests)")

# %% Initialize connections
con = duckdb.connect("311_equity_demographics.duckdb")
print("DuckDB connection established")

phl = Philly()
print("Philly API initialized")

# %% Configuration
CONFIG = {
    "years": [2023, 2024],
    "min_requests_per_tract": 10,  # Filter tracts with fewer requests
    "correlation_method": "spearman",  # or 'pearson'
    "significance_level": 0.05,
}

print("\nAnalysis Configuration:")
print(f"  - Years: {CONFIG['years']}")
print(f"  - Minimum requests per tract: {CONFIG['min_requests_per_tract']}")
print(f"  - Correlation method: {CONFIG['correlation_method']}")

# %% [markdown]
# ## Part 1: Load 311 Service Request Data

# %% Load 311 requests with error handling
print("\n" + "=" * 70)
print("PART 1: LOADING 311 SERVICE REQUESTS")
print("=" * 70)


def load_311_data(years: list[int]) -> bool:
    """Load 311 data for specified years with robust error handling."""
    try:
        urls = []
        for year in years:
            url = f"https://phl.carto.com/api/v2/sql?filename=public_cases_fc&format=csv&skipfields=cartodb_id,the_geom,the_geom_webmercator&q=SELECT * FROM public_cases_fc WHERE requested_datetime >= '{year}-01-01' AND requested_datetime < '{year + 1}-01-01'"
            urls.append(url)

        print(f"Loading data for years: {years}...")

        # Build UNION query for all years
        union_parts = " UNION ALL ".join(
            ["SELECT * FROM read_csv_auto(?)" for _ in urls]
        )
        con.execute(
            f"CREATE OR REPLACE TABLE service_311 AS {union_parts}",
            urls,
        )

        # Validate data loaded
        result = con.execute("SELECT COUNT(*) as count FROM service_311").fetchdf()
        count = result["count"].iloc[0]

        if count == 0:
            print("⚠️  WARNING: No 311 data loaded! Check data availability.")
            return False

        print(f"✓ Loaded {count:,} 311 service requests")

        # Show sample
        sample = con.execute("SELECT * FROM service_311 LIMIT 3").fetchdf()
        if len(sample) > 0:
            print("\nSample columns:", list(sample.columns[:8]), "...")

        return True

    except Exception as e:
        print(f"❌ Error loading 311 data: {e}")
        return False


data_loaded = load_311_data(CONFIG["years"])

if not data_loaded:
    print("\n⚠️  Cannot proceed without 311 data. Exiting.")
    sys.exit(1)

# %% Calculate response times
print("\nCalculating response time metrics...")

try:
    con.execute("""
    CREATE OR REPLACE TABLE service_311_with_metrics AS
    SELECT *,
        TRY_CAST(requested_datetime AS TIMESTAMP) as requested_ts,
        TRY_CAST(updated_datetime AS TIMESTAMP) as updated_ts,
        TRY_CAST(closed_datetime AS TIMESTAMP) as closed_ts,
        DATEDIFF('hour',
            TRY_CAST(requested_datetime AS TIMESTAMP),
            TRY_CAST(updated_datetime AS TIMESTAMP)
        ) as response_hours,
        DATEDIFF('day',
            TRY_CAST(requested_datetime AS TIMESTAMP),
            TRY_CAST(closed_datetime AS TIMESTAMP)
        ) as resolution_days
    FROM service_311
    WHERE requested_datetime IS NOT NULL
    """)

    # Basic statistics
    stats_result = con.execute("""
    SELECT
        COUNT(*) as total_requests,
        COUNT(DISTINCT service_name) as unique_service_types,
        COUNT(CASE WHEN status = 'Closed' THEN 1 END) as closed_requests,
        ROUND(COUNT(CASE WHEN status = 'Closed' THEN 1 END)::FLOAT / COUNT(*)::FLOAT * 100, 2) as pct_closed,
        AVG(response_hours) as avg_response_hours,
        MEDIAN(response_hours) as median_response_hours
    FROM service_311_with_metrics
    WHERE response_hours >= 0 AND response_hours < 8760
    """).fetchdf()

    print("\n311 SERVICE STATISTICS:")
    print(stats_result.to_string(index=False))

except Exception as e:
    print(f"❌ Error calculating metrics: {e}")
    sys.exit(1)

# %% [markdown]
# ## Part 2: Load Census Tracts and Demographics

# %% Load census tracts
print("\n" + "=" * 70)
print("PART 2: LOADING CENSUS TRACTS AND DEMOGRAPHICS")
print("=" * 70)


async def load_census_data():
    """Load census tract boundaries from OpenDataPhilly."""
    try:
        print("Loading census tract boundaries...")
        tracts = await phl.load("Census Tracts", format="geojson")

        if not tracts or "features" not in tracts:
            print("⚠️  WARNING: No census tract data received")
            return None

        gdf = gpd.GeoDataFrame.from_features(tracts["features"])
        gdf = gdf.set_crs("EPSG:4326")

        print(f"✓ Loaded {len(gdf)} census tracts")
        return gdf

    except Exception as e:
        print(f"❌ Error loading census tracts: {e}")
        return None


census_tracts = asyncio.run(load_census_data())

if census_tracts is None:
    print("\n⚠️  Cannot proceed without census tracts. Exiting.")
    sys.exit(1)

# %% Load demographic data from Census ACS API
print("\nLoading demographic data from Census Bureau ACS 5-Year Estimates...")

try:
    import json
    import subprocess

    # Fetch Census ACS data using curl
    # B19013_001E: Median Household Income
    # B17001_001E: Total population for poverty calculation
    # B17001_002E: Population Below Poverty Level
    # B23025_003E: Unemployed
    # B25077_001E: Median Home Value

    census_url = "https://api.census.gov/data/2022/acs/acs5?get=NAME,B19013_001E,B17001_001E,B17001_002E,B23025_003E,B25077_001E&for=tract:*&in=state:42&in=county:101"

    print("Fetching data from Census Bureau API...")
    result = subprocess.run(["curl", "-s", census_url], capture_output=True, text=True)

    if result.returncode == 0:
        census_data = json.loads(result.stdout)

        # First row is headers
        headers = census_data[0]
        rows = census_data[1:]

        demographics = pd.DataFrame(rows, columns=headers)

        # Convert numeric columns
        demographics["median_income"] = pd.to_numeric(
            demographics["B19013_001E"], errors="coerce"
        )
        demographics["population"] = pd.to_numeric(
            demographics["B17001_001E"], errors="coerce"
        )
        demographics["pop_in_poverty"] = pd.to_numeric(
            demographics["B17001_002E"], errors="coerce"
        )
        demographics["unemployed"] = pd.to_numeric(
            demographics["B23025_003E"], errors="coerce"
        )
        demographics["median_home_value"] = pd.to_numeric(
            demographics["B25077_001E"], errors="coerce"
        )

        # Calculate poverty rate
        demographics["poverty_rate"] = (
            demographics["pop_in_poverty"] / demographics["population"] * 100
        )

        # Create GEOID10 format to match census tracts (state + county + tract)
        demographics["GEOID10"] = (
            demographics["state"] + demographics["county"] + demographics["tract"]
        )

        # Replace Census Bureau missing data flag (-666666666) with NaN
        demographics["median_home_value"] = demographics["median_home_value"].replace(
            -666666666, np.nan
        )

        print(f"✓ Loaded demographics for {len(demographics)} census tracts")
        print("\nDemographic Data Summary:")
        print(
            f"  Median Income Range: ${demographics['median_income'].min():,.0f} - ${demographics['median_income'].max():,.0f}"
        )
        print(
            f"  Poverty Rate Range: {demographics['poverty_rate'].min():.1f}% - {demographics['poverty_rate'].max():.1f}%"
        )
        print(f"  Mean Poverty Rate: {demographics['poverty_rate'].mean():.1f}%")

        HAS_DEMOGRAPHICS = True
    else:
        print(f"❌ Error fetching census data: {result.stderr}")
        HAS_DEMOGRAPHICS = False
        demographics = None

except Exception as e:
    print(f"❌ Demographics loading failed: {e}")
    import traceback

    traceback.print_exc()
    HAS_DEMOGRAPHICS = False
    demographics = None

# %% [markdown]
# ## Part 3: Spatial Analysis - Join 311 to Census Tracts

# %% Prepare for spatial join
print("\n" + "=" * 70)
print("PART 3: SPATIAL ANALYSIS")
print("=" * 70)

print("Preparing 311 data for spatial join...")

try:
    # Load with valid coordinates and reasonable bounds
    service_df = con.execute("""
    SELECT *
    FROM service_311_with_metrics
    WHERE lat IS NOT NULL
        AND lon IS NOT NULL
        AND TRY_CAST(lat AS DOUBLE) IS NOT NULL
        AND TRY_CAST(lon AS DOUBLE) IS NOT NULL
        AND TRY_CAST(lat AS DOUBLE) BETWEEN 39.8 AND 40.2
        AND TRY_CAST(lon AS DOUBLE) BETWEEN -75.3 AND -74.9
    LIMIT 500000
    """).fetchdf()

    if len(service_df) == 0:
        print("❌ No valid geocoded 311 requests found!")
        sys.exit(1)

    print(f"✓ {len(service_df):,} 311 requests with valid coordinates")

    # Convert to GeoDataFrame
    service_gdf = gpd.GeoDataFrame(
        service_df,
        geometry=gpd.points_from_xy(
            pd.to_numeric(service_df.lon, errors="coerce"),
            pd.to_numeric(service_df.lat, errors="coerce"),
        ),
        crs="EPSG:4326",
    )

    # Spatial join
    print("Performing spatial join...")
    service_by_tract = gpd.sjoin(
        service_gdf,
        census_tracts[["geometry", "GEOID10", "NAME10"]].reset_index(),
        how="left",
        predicate="within",
    )

    matched = service_by_tract["GEOID10"].notna().sum()
    match_pct = matched / len(service_by_tract) * 100

    print(f"✓ Matched {matched:,} requests to tracts ({match_pct:.1f}%)")

except Exception as e:
    print(f"❌ Error in spatial join: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# %% Aggregate by census tract
print("\nAggregating metrics by census tract...")

tract_metrics = (
    service_by_tract.groupby("GEOID10")
    .agg(
        {
            "service_request_id": "count",
            "response_hours": ["median", "mean"],
            "resolution_days": ["median", "mean"],
            "status": lambda x: (x == "Closed").sum(),
        }
    )
    .reset_index()
)

# Flatten column names
tract_metrics.columns = [
    "GEOID10",
    "total_requests",
    "median_response_hours",
    "mean_response_hours",
    "median_resolution_days",
    "mean_resolution_days",
    "closed_requests",
]

# Calculate resolution rate
tract_metrics["resolution_rate"] = (
    tract_metrics["closed_requests"] / tract_metrics["total_requests"] * 100
)

# Filter to tracts with minimum requests
tract_metrics_filtered = tract_metrics[
    tract_metrics["total_requests"] >= CONFIG["min_requests_per_tract"]
].copy()

print(f"✓ Calculated metrics for {len(tract_metrics_filtered)} census tracts")
print(f"  (filtered to tracts with >= {CONFIG['min_requests_per_tract']} requests)")

# %% Join demographics with tract metrics
if HAS_DEMOGRAPHICS and demographics is not None:
    print("\nJoining demographic data with 311 service metrics...")

    # Merge on GEOID10
    tract_metrics_with_demos = tract_metrics_filtered.merge(
        demographics[
            [
                "GEOID10",
                "median_income",
                "poverty_rate",
                "population",
                "median_home_value",
                "unemployed",
            ]
        ],
        on="GEOID10",
        how="left",
    )

    # Calculate per-capita metrics
    tract_metrics_with_demos["requests_per_1000"] = (
        tract_metrics_with_demos["total_requests"]
        / tract_metrics_with_demos["population"]
        * 1000
    )

    matched = tract_metrics_with_demos["median_income"].notna().sum()
    total = len(tract_metrics_with_demos)

    print(
        f"✓ Joined demographics for {matched}/{total} tracts ({matched / total * 100:.1f}%)"
    )
    print("\nJoined Data Summary:")
    print(f"  Tracts with income data: {matched}")
    print(
        f"  Income range: ${tract_metrics_with_demos['median_income'].min():,.0f} - ${tract_metrics_with_demos['median_income'].max():,.0f}"
    )
    print(
        f"  Requests per 1,000 residents: {tract_metrics_with_demos['requests_per_1000'].mean():.1f} (avg)"
    )

    # Remove tracts with missing income data for correlation analysis
    tract_metrics_complete = tract_metrics_with_demos.dropna(
        subset=["median_income", "median_response_hours"]
    )
    print(f"  Tracts with complete data for correlation: {len(tract_metrics_complete)}")
else:
    tract_metrics_with_demos = tract_metrics_filtered
    tract_metrics_complete = None

# %% [markdown]
# ## Part 4: Statistical Analysis

# %% Correlation analysis (if demographics available)
print("\n" + "=" * 70)
print("PART 4: STATISTICAL ANALYSIS")
print("=" * 70)

if (
    HAS_DEMOGRAPHICS
    and tract_metrics_complete is not None
    and len(tract_metrics_complete) > 0
):
    print("\n📊 INCOME vs. RESPONSE TIME CORRELATION ANALYSIS\n")

    # Calculate Spearman correlation (better for non-linear relationships)
    spearman_corr, spearman_p = stats.spearmanr(
        tract_metrics_complete["median_income"],
        tract_metrics_complete["median_response_hours"],
    )

    # Calculate Pearson correlation (assumes linear relationship)
    pearson_corr, pearson_p = stats.pearsonr(
        tract_metrics_complete["median_income"],
        tract_metrics_complete["median_response_hours"],
    )

    print(f"Sample Size: {len(tract_metrics_complete)} census tracts\n")

    print("SPEARMAN CORRELATION (Rank-based, robust to outliers):")
    print(f"  Correlation coefficient: {spearman_corr:.4f}")
    print(f"  P-value: {spearman_p:.6f}")
    print(f"  Significant at α=0.05? {'YES ✓' if spearman_p < 0.05 else 'NO ✗'}")

    print("\nPEARSON CORRELATION (Linear relationship):")
    print(f"  Correlation coefficient: {pearson_corr:.4f}")
    print(f"  P-value: {pearson_p:.6f}")
    print(f"  Significant at α=0.05? {'YES ✓' if pearson_p < 0.05 else 'NO ✗'}")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if spearman_corr < 0:
        print(f"  ⚠️  NEGATIVE correlation detected: {spearman_corr:.4f}")
        print("  → As income INCREASES, response time DECREASES (faster service)")
        print("  → As income DECREASES, response time INCREASES (slower service)")
        if spearman_p < 0.05:
            print("  → This correlation is STATISTICALLY SIGNIFICANT (p < 0.05)")
            print("  → This suggests SYSTEMIC INEQUITY in city services")
        else:
            print("  → Not statistically significant (p >= 0.05)")
    elif spearman_corr > 0:
        print(f"  Positive correlation: {spearman_corr:.4f}")
        print("  → As income INCREASES, response time also INCREASES")
        print("  → This would suggest LOWER-income areas get FASTER service")
        if spearman_p < 0.05:
            print("  → This correlation is statistically significant (p < 0.05)")
    else:
        print("  No meaningful correlation detected")

    # Poverty rate vs response time
    print("\n📊 POVERTY RATE vs. RESPONSE TIME CORRELATION\n")

    poverty_complete = tract_metrics_complete.dropna(subset=["poverty_rate"])
    if len(poverty_complete) > 0:
        poverty_spearman, poverty_p = stats.spearmanr(
            poverty_complete["poverty_rate"], poverty_complete["median_response_hours"]
        )

        print(f"Sample Size: {len(poverty_complete)} census tracts")
        print(f"  Spearman correlation: {poverty_spearman:.4f}")
        print(f"  P-value: {poverty_p:.6f}")
        print(f"  Significant at α=0.05? {'YES ✓' if poverty_p < 0.05 else 'NO ✗'}")

        if poverty_spearman > 0 and poverty_p < 0.05:
            print("\n  ⚠️  POSITIVE correlation: Higher poverty = Slower response times")
            print(
                "  → This confirms service inequity affecting disadvantaged communities"
            )

    # Compare extremes
    print("\n📊 COMPARING INCOME EXTREMES\n")

    # Split into quintiles
    tract_metrics_complete["income_quintile"] = pd.qcut(
        tract_metrics_complete["median_income"],
        q=5,
        labels=["Lowest 20%", "Low", "Middle", "High", "Highest 20%"],
    )

    quintile_stats = (
        tract_metrics_complete.groupby("income_quintile")
        .agg(
            {
                "median_response_hours": ["median", "mean"],
                "median_income": "median",
                "total_requests": "sum",
            }
        )
        .round(1)
    )

    print("Response Times by Income Quintile:")
    print(quintile_stats)

    lowest_quintile = tract_metrics_complete[
        tract_metrics_complete["income_quintile"] == "Lowest 20%"
    ]
    highest_quintile = tract_metrics_complete[
        tract_metrics_complete["income_quintile"] == "Highest 20%"
    ]

    lowest_median = lowest_quintile["median_response_hours"].median()
    highest_median = highest_quintile["median_response_hours"].median()
    gap = lowest_median - highest_median

    print("\nKEY FINDING:")
    print(
        f"  Lowest-income areas (bottom 20%): {lowest_median:.1f} hours median response"
    )
    print(
        f"  Highest-income areas (top 20%): {highest_median:.1f} hours median response"
    )
    print(f"  GAP: {gap:.1f} hours ({gap / 24:.1f} days)")

    if gap > 0:
        print(f"  → Low-income areas wait {gap:.1f} hours LONGER")
        print("  → This represents measurable service inequity")

else:
    print("\n⚠️  Skipping income correlation (no demographic data available)")
    print("This is a MAJOR LIMITATION of the current analysis.")

# %% Basic disparity analysis (income-independent)
print("\n📊 Response Time Disparity Analysis (Geographic Only)")

summary = {
    "Total Tracts Analyzed": len(tract_metrics_filtered),
    "Median Response Time": f"{tract_metrics_filtered['median_response_hours'].median():.1f} hours",
    "Fastest Tract": f"{tract_metrics_filtered['median_response_hours'].min():.1f} hours",
    "Slowest Tract": f"{tract_metrics_filtered['median_response_hours'].max():.1f} hours",
    "Disparity Range": f"{tract_metrics_filtered['median_response_hours'].max() - tract_metrics_filtered['median_response_hours'].min():.1f} hours",
    "Std Deviation": f"{tract_metrics_filtered['median_response_hours'].std():.1f} hours",
}

for key, value in summary.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Part 5: Visualizations

# %% Create visualizations
if (
    HAS_DEMOGRAPHICS
    and tract_metrics_complete is not None
    and len(tract_metrics_complete) > 0
):
    print("\n" + "=" * 70)
    print("PART 5: CREATING VISUALIZATIONS")
    print("=" * 70)

    # 1. Scatter plot: Income vs Response Time
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(
        tract_metrics_complete["median_income"] / 1000,
        tract_metrics_complete["median_response_hours"],
        alpha=0.6,
        s=50,
    )
    plt.xlabel("Median Household Income ($1000s)", fontsize=12)
    plt.ylabel("Median Response Time (hours)", fontsize=12)
    plt.title(
        "Income vs 311 Response Time\nPhiladelphia Census Tracts (2023-2024)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Add correlation text
    corr_text = f"Spearman ρ = {spearman_corr:.3f}\np = {spearman_p:.4f}"
    plt.text(
        0.05,
        0.95,
        corr_text,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        verticalalignment="top",
        fontsize=10,
    )

    # 2. Box plot by income quintile
    plt.subplot(1, 2, 2)
    tract_metrics_complete.boxplot(
        column="median_response_hours", by="income_quintile", ax=plt.gca()
    )
    plt.xlabel("Income Quintile", fontsize=12)
    plt.ylabel("Median Response Time (hours)", fontsize=12)
    plt.title("Response Times by Income Level", fontsize=14, fontweight="bold")
    plt.suptitle("")  # Remove default title
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("income_vs_response_time.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: income_vs_response_time.png")

    # 3. Create choropleth map
    print("\nCreating choropleth maps...")

    # Merge with census tract geometries
    census_with_metrics = census_tracts.merge(
        tract_metrics_with_demos, on="GEOID10", how="left"
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Map 1: Median Income
    ax1 = axes[0]
    census_with_metrics.plot(
        column="median_income",
        ax=ax1,
        legend=True,
        cmap="RdYlGn",
        edgecolor="black",
        linewidth=0.3,
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": "Median Income ($)", "orientation": "horizontal"},
    )
    ax1.set_title(
        "Median Household Income by Census Tract\nPhiladelphia (2022 ACS)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.axis("off")

    # Map 2: Response Time
    ax2 = axes[1]
    census_with_metrics.plot(
        column="median_response_hours",
        ax=ax2,
        legend=True,
        cmap="RdYlBu_r",  # Reversed: red = slow, blue = fast
        edgecolor="black",
        linewidth=0.3,
        missing_kwds={"color": "lightgrey"},
        legend_kwds={"label": "311 Response Time (hours)", "orientation": "horizontal"},
    )
    ax2.set_title(
        "311 Service Response Times by Census Tract\nPhiladelphia (2023-2024)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig("maps_income_and_response.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: maps_income_and_response.png")

    # 4. Poverty rate vs response time
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(
        tract_metrics_complete["poverty_rate"],
        tract_metrics_complete["median_response_hours"],
        alpha=0.6,
        s=50,
        c=tract_metrics_complete["median_income"],
        cmap="viridis",
    )
    plt.colorbar(label="Median Income ($)")
    plt.xlabel("Poverty Rate (%)", fontsize=12)
    plt.ylabel("Median Response Time (hours)", fontsize=12)
    plt.title(
        "Poverty Rate vs 311 Response Time\nPhiladelphia Census Tracts (2023-2024)",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("poverty_vs_response_time.png", dpi=150, bbox_inches="tight")
    print("✓ Saved: poverty_vs_response_time.png")

    print("\n✓ All visualizations created successfully")

# %% [markdown]
# ## Part 6: Save Results

# %% Export results
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

try:
    # Save to database - use enhanced data if available
    if HAS_DEMOGRAPHICS and tract_metrics_with_demos is not None:
        data_to_save = tract_metrics_with_demos
        csv_filename = "tract_metrics_with_demographics.csv"
    else:
        data_to_save = tract_metrics_filtered
        csv_filename = "tract_metrics_basic.csv"

    con.register("tract_metrics_df", data_to_save)
    con.execute(
        "CREATE OR REPLACE TABLE tract_service_metrics AS SELECT * FROM tract_metrics_df"
    )

    # Export CSVs
    data_to_save.to_csv(csv_filename, index=False)

    print("✓ Saved to database: 311_equity_demographics.duckdb")
    print(f"✓ Exported: {csv_filename}")
    print(f"  - {len(data_to_save)} census tracts")
    print(f"  - {len(data_to_save.columns)} columns")

except Exception as e:
    print(f"⚠️  Error saving results: {e}")

# Close connection
con.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

if HAS_DEMOGRAPHICS:
    print("\n✓ COMPLETE EQUITY ANALYSIS SUCCESSFUL")
    print("  → Census income data integrated from ACS 5-Year Estimates")
    print("  → Statistical correlations calculated")
    print("  → Visualizations generated")
    print("  → Ready for publication")
else:
    print("\n⚠️  IMPORTANT LIMITATION:")
    print(
        "This analysis could not test income-based disparities due to lack of demographic data."
    )
    print("The investigation remains incomplete without this critical component.")

print("\n✓ Investigation script completed successfully")
