# (philly) cheesesnake

<img src="./assets/logo.png" width="256" alt="logo">

Python multitool and unified query layer (built on [DuckDB](https://duckdb.org/)) for working with [OpenDataPhilly](https://opendataphilly.org/) datasets!

## Examples

```python
from cheesesnake import Cheesesnake

cn = Cheesesnake()

cn.query_datasets("SELECT title FROM datasets WHERE LOWER(title) LIKE '%septa%'")
```
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SEPTA Alerts</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SEPTA Bus Detours</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SEPTA Elevator Outages</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SEPTA Finances</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SEPTA GTFS</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SEPTA GTFS Real-time Alerts and Updates</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SEPTA Performance Dashboards</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SEPTA Real-time Map</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SEPTA Regional Rail APIs</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SEPTA Regional Rail On-time Performance Reports</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SEPTA Ridership Statistics</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SEPTA Routes, Stops, and Locations</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SEPTA SMS Transit</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SEPTA Schedules</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SEPTA TransitView</td>
    </tr>
    <tr>
      <th>15</th>
      <td>SEPTA Trip Planner</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Skookul SEPTA Real Time Locator</td>
    </tr>
  </tbody>
</table>
</div>

## Update Datasets

```bash
uv run scripts/update_datasets.py
```