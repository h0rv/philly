# Exploration Ideas

## 1. Every Block Has a Changelog

A simple, high-concept exploration: enter any Philadelphia address and see the recent "change history" of that block, like a city-scale activity feed.

### Core idea

Turn open data into a block-by-block timeline of what changed nearby:

- new business licenses
- 311 requests opened / closed
- crime incidents
- demolitions
- lane closures
- permits or other city activity

### Why it works

- extremely easy to explain
- feels fresh and magical
- personal and local without being overly political
- showcases why `philly` is useful: discover, filter, stream, join, and summarize many datasets fast

### Candidate datasets

- `311 Service and Information Requests`
- `Crime Incidents`
- `Licenses and Inspections Business Licenses`
- `Building Demolitions`
- `Street Lane Closures`
- `Licenses and Inspections Complaints`
- `Vacant Property Indicators`
- `Census Blocks`

### UX shape

- address search
- "last 30 / 90 / 365 days" toggle
- reverse chronological event feed
- summary stats for the selected block
- comparison against citywide baseline

---

## 2. A Changelog for Philadelphia Blocks

A more opinionated framing of the changelog idea: model each block as a place with bursts of activity, quiet periods, and a running history of visible change.

### Core idea

Represent each meaningful event as a simple changelog entry:

- `opening:` new business or permit
- `resolved:` 311 issue resolved
- `incident:` major crash / demolition / repeated complaints
- `routine:` scheduled maintenance / cleaning / lane closure

### Why it works

- memorable metaphor
- likely to resonate with developers, Hacker News, and agent/tooling audiences
- naturally demonstrates how city data can become programmable state changes

### Candidate datasets

- everything in the changelog idea, especially:
- `Citywide Cleaning Program Tasks`
- `Traffic Calming Devices`
- `Licenses and Inspections Code Violations`
- `Street's Code Violation Notices`
- `PPR Hydration Stations` or `Philadelphia Tree Inventory` for neighborhood context

### UX shape

- changelog feed with event types and colors
- "most active blocks this week"
- "quietest stable blocks"
- compare one block in 2024 vs 2025

---

## 3. Philly Rate of Change

A citywide map of where Philadelphia is changing fastest right now.

### Core idea

Instead of showing one issue, combine multiple signals into a single "change velocity" score:

- new business activity
- demolitions
- 311 volume shifts
- crime change vs prior period
- lane closures / traffic interventions
- vacancies or cleanup activity

### Why it works

- broad, visual, and intuitive
- underexplored compared to single-topic maps
- useful for residents, journalists, planners, and curious outsiders
- a strong showcase of multi-dataset fusion

### Candidate datasets

- `Licenses and Inspections Business Licenses`
- `Building Demolitions`
- `311 Service and Information Requests`
- `Crime Incidents`
- `Street Lane Closures`
- `Traffic Calming Devices`
- `Vacant Lot Cleanups`
- `Citywide Cleaning Program Tasks`
- `Census Blocks` or `Census Tracts`

### Possible outputs

- heatmap of fast-changing blocks / tracts
- ranked list of "hottest" corridors
- filter by change type: safety, development, services, street life
- time slider for month-over-month / year-over-year change

---

# Dataset Landscape and Composition Notes

## Catalog snapshot

Broad pass over the packaged catalog in this repo:

- **412 datasets** in `src/philly/datasets`
- **289** have CSV resources
- **284** have geospatial resources (`geojson` or `shp`)
- **70** have meaningful multi-year or repeated snapshot history
- **47** look current / latest / ongoing / daily / real-time-ish
- **53** are especially valuable because they are both **geospatial and longitudinal**

This means the strongest opportunities are not just single datasets, but compositions of:

- event data
- geography layers
- historical snapshots
- infrastructure layers
- registry / administrative data

## Main dataset clusters

### 1. City operations / urban telemetry

Operational datasets that make the city feel alive:

- `311 Service and Information Requests`
- `Crime Incidents`
- `Building Demolitions`
- `Citywide Cleaning Program Tasks`
- `Street Lane Closures`
- `Traffic Calming Devices`
- `Big Belly Trash Bin Usage`
- `AMS Latest Air Quality Sensor Readings`
- `Free Wi-Fi Locations`

### 2. Longitudinal change datasets

Strong for rate-of-change and before/after analysis:

- `Crime Incidents`
- `Crashes data`
- `Parking Violations`
- `Real Estate Transfers`
- `Philadelphia Tree Inventory`
- `PPR Tree Canopy`
- `Large Building Energy Benchmarking Data`
- `SEPTA Ridership Statistics`
- `Affordable Housing Production`
- `Litter Index`
- `Campaign Finance Reports`

### 3. Property / parcel / building state datasets

Strongest area for entity joins and timelines:

- `Real Estate Transfers`
- `Department of Records Property Parcels`
- `Vacant Property Indicators`
- `Vacant Property Indicators Percentage by Block`
- `LandCare Program`
- `Licenses and Inspections Business Licenses`
- `Licenses and Inspections Building and Zoning Permits`
- `Licenses and Inspections Complaints`
- `Licenses and Inspections Code Violations`
- `Licenses and Inspections: Case Investigations`
- `Affordable Housing Production`
- `Lead Paint Certifications`
- `Certified for Rental Suitability`
- `Real Estate Tax Balances`

### 4. Street / segment / node / safety graph

Strong for network analysis and safety intervention stories:

- `Street Centerlines`
- `Traffic Calming Devices`
- `Street Lane Closures`
- `Vision Zero High Injury Network`
- `Crashes data`
- `School Crossing Guards Locations`
- `Bus Shelters`
- `SEPTA Routes, Stops, and Locations`
- `SEPTA Ridership Statistics`
- `Red Light Cameras`

### 5. Climate / resilience / public realm

Good for practical, high-signal, less-explored civic products:

- `Heat Vulnerability by Census Tract`
- `Philadelphia Tree Inventory`
- `PPR Tree Canopy`
- `PPR Hydration Stations`
- `FEMA Flood Plain`
- `Green Stormwater Infrastructure *`
- `Impervious Surfaces`
- `Water Inlets`
- `Rain Check Installation Sites`
- `LandCare Program`
- `Air Monitoring Stations`
- `AMS Latest Air Quality Sensor Readings`

### 6. Digital access / public access

Unexpectedly rich:

- `Free Wi-Fi Locations`
- `Philly KEYSPOT Locations`
- `Philadelphia Household Internet Assessment Survey`

### 7. Economy / procurement / business graph

Strong for business ecosystem and city-as-customer compositions:

- `City-Registered Local Businesses`
- `Licenses and Inspections Business Licenses`
- `Professional Services Contracts`
- `Commodities Contracts`
- `City Payments`
- `Commercial Corridors of Philadelphia`
- `Business Improvement Districts (BID)`
- `Boost Your Business Program`
- `Catalyst Fund Grants`
- `Instore Forgivable Loan Program`

## Strong join keys and composition surfaces

### 1. Census geography

Very common and useful join surface:

- `GEOID10`
- census tract IDs
- block IDs
- block group IDs
- ZIP / ward / council district / planning district

Useful for linking:

- heat
- food access
- internet access
- crime
- 311
- ridership
- vacancy
- health indicators

### 2. Parcel / property identity

The richest entity graph in the catalog:

- `opa_account_num`
- `brt_id`
- `parcel_id_num`
- `pin`
- `reg_map_id`
- standardized address fields

This supports a true property-state-machine style product.

### 3. Street segment / node IDs

Very underexplored and high value:

- `seg_id`
- `fnode_`
- `tnode_`
- `node_id`

This enables street-level joins without fuzzy matching.

### 4. Transit stop IDs

Useful for pedestrian / transit analysis:

- `Stop_Code`
- `Stop_ID`
- `stopid`

### 5. Vendor / business name overlap

Messier, but real and usable after normalization.

Observed normalized overlap:

- `City-Registered Local Businesses` ∩ `Professional Services Contracts`: **33** names
- `City-Registered Local Businesses` ∩ `City Payments`: **62** names

Examples include:

- `BICYCLE TRANSIT SYSTEMS`
- `ECONSULT SOLUTIONS`
- `ELLIOTT LEWIS`
- `DANIEL J KEATING`
- `CEISLER MEDIA ISSUE ADVOCACY`

This suggests a real city spending / local vendor graph is possible.

## Hidden gems and underexplored datasets

### `Free Wi-Fi Locations`

Not just a point layer. It already includes tract-level enrichment like:

- income
- internet access
- device access
- broadband access
- public Wi-Fi status
- fiber / Meraki / speed fields

### `Big Belly Trash Bin Usage`

An unusual event/telemetry dataset with timestamps and fill-state signals.
Potential for trash pressure, corridor intensity, and nightlife / demand proxies.

### `Citywide Cleaning Program Tasks`

Rare operational cleanup dataset with task names and timestamps.
Good for maintenance metabolism, before/after cleaning, and sanitation demand work.

### `Traffic Calming Devices`

Contains install dates, making real before/after safety analysis possible.

### `Vacant Property Indicators Percentage by Block`

A useful pre-aggregated block-level vacancy layer, updated recently.

### `PPR Tree Canopy`

Especially valuable for its **2008-2018 canopy gain/loss** layer.

### `SEPTA Ridership Statistics`

Much richer than a transit map: stop-level use, route summaries, district summaries, and time-series behavior.

## Compositional opportunities

### A. Property lifecycle graph

Combine:

- parcels
- transfers
- permits
- complaints
- violations
- demolitions
- affordable housing
- vacancy indicators
- LandCare
- tax balances

Result:

- every parcel becomes a timeline / state machine

### B. Street intervention effectiveness

Combine:

- crashes
- Vision Zero high injury network
- traffic calming devices
- school crossing guards
- bus shelters
- SEPTA stop ridership
- lane closures
- street centerlines

Result:

- a street-level map of risk, intervention, and coverage

### C. Heat refuge / summer survivability

Combine:

- heat vulnerability
- tree canopy change
- tree inventory
- hydration stations
- playgrounds
- bus shelters
- air monitoring
- free Wi-Fi / public indoor-ish access
- parks / rec sites

Result:

- a practical map of where people can actually find shade, water, cooling, and refuge

### D. Cleanliness / maintenance metabolism

Combine:

- Big Belly usage
- Citywide Cleaning Program Tasks
- Litter Index
- waste basket locations
- 311 sanitation requests
- vacant lot cleanups
- business corridors

Result:

- a living map of where the city fills up, gets dirty, and recovers fastest

### E. Digital access reality map

Combine:

- Free Wi-Fi Locations
- Philly KEYSPOT Locations
- Household Internet Assessment Survey
- schools / rec centers / libraries / parks

Result:

- a useful map of where people can actually get online versus where need is highest

### F. The city as customer

Combine:

- City-Registered Local Businesses
- City Payments
- Professional Services Contracts
- Commodities Contracts
- Business Licenses
- Commercial Corridors

Result:

- a new derived dataset showing which local businesses are actually in the city’s economic orbit

## Strongest raw ingredients for future explorations

### Best operational datasets

- `311 Service and Information Requests`
- `Crime Incidents`
- `Building Demolitions`
- `Street Lane Closures`
- `Traffic Calming Devices`
- `Citywide Cleaning Program Tasks`

### Best base geography

- `Census Blocks`
- `Census Tracts`
- `Street Centerlines`
- `Department of Records Property Parcels`

### Best enrichment layers

- `Heat Vulnerability by Census Tract`
- `Philadelphia Tree Inventory`
- `PPR Tree Canopy`
- `Vacant Property Indicators Percentage by Block`
- `Neighborhood Food Retail`
- `Free Wi-Fi Locations`
- `SEPTA Ridership Statistics`

### Best business / registry layers

- `City-Registered Local Businesses`
- `Licenses and Inspections Business Licenses`
- `Professional Services Contracts`
- `City Payments`
- `Affordable Housing Production`
- `Real Estate Transfers`
