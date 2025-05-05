# (philly) cheesesnake

<img src="./assets/logo.png" width="256" alt="logo">

Python multitool and unified query layer (built on [DuckDB](https://duckdb.org/)) for working with [OpenDataPhilly](https://opendataphilly.org/) datasets!

## Examples

```python
from cheesesnake import Cheesesnake

cs = Cheesesnake()

cs.query("SELECT title FROM datasets WHERE LOWER(title) LIKE '%septa%'")
"""
┌─────────────────────────────────────────────────┐
│                      title                      │
│                     varchar                     │
├─────────────────────────────────────────────────┤
│ SEPTA Alerts                                    │
│ SEPTA Bus Detours                               │
│ SEPTA Elevator Outages                          │
│ SEPTA Finances                                  │
│ SEPTA GTFS                                      │
│ SEPTA GTFS Real-time Alerts and Updates         │
│ SEPTA Performance Dashboards                    │
│ SEPTA Real-time Map                             │
│ SEPTA Regional Rail APIs                        │
│ SEPTA Regional Rail On-time Performance Reports │
│ SEPTA Ridership Statistics                      │
│ SEPTA Routes, Stops, and Locations              │
│ SEPTA SMS Transit                               │
│ SEPTA Schedules                                 │
│ SEPTA TransitView                               │
│ SEPTA Trip Planner                              │
│ Skookul SEPTA Real Time Locator                 │
├─────────────────────────────────────────────────┤
│                     17 rows                     │
└─────────────────────────────────────────────────┘
"""
```

Query all resources:

```sql
SELECT
  unnest.name,
  unnest.format,
  unnest.url
FROM datasets,
UNNEST(resources) AS unnest;
```

## Update Datasets

```bash
uv run scripts/update_datasets.py
```

## Resources

* OpenDataPhilly
    * https://opendataphilly.org/
    * https://github.com/opendataphilly/
