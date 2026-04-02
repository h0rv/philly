export const explorations = [
  {
    slug: 'philly-timelapse',
    title: 'City Atlas',
    summary:
      'A full-screen timelapse that scrubs Philadelphia from 1860 atlas plates to modern aerial imagery while holding the same camera over the city.',
    tags: ['Maps', 'History', 'Aerial'],
    featured: true,
    requiredFiles: ['index.html', 'sw.js'],
    publishFiles: ['index.html', 'sw.js'],
    buildHint: 'Static artifact already checked in under explorations/philly-timelapse.',
  },
  {
    slug: 'block-report-card',
    title: "Your Block's Report Card",
    summary:
      'Type an address or click the map to see a neighborhood score built from crime, 311 response times, parking violations, and litter data.',
    tags: ['Interactive', 'deck.gl', 'Geocoding'],
    featured: true,
    requiredFiles: ['index.html', 'data/blocks.geojson', 'data/grades.json'],
    publishFiles: ['index.html', 'data/blocks.geojson', 'data/grades.json'],
    buildHint: 'uv run python explorations/block-report-card/fetch_data.py',
  },
  {
    slug: 'city-pulse',
    title: 'Philadelphia Bike Flows',
    summary:
      'A 24-hour Indego flow map that shows the city reversing course across the workday, from University City into Center City and back again.',
    tags: ['Animation', 'MapLibre', 'Indego'],
    featured: true,
    requiredFiles: ['index.html', 'data/bike_flows.json'],
    publishFiles: ['index.html', 'data/bike_flows.json'],
    buildHint: 'python explorations/city-pulse/process_trips.py',
  },
  {
    slug: 'property-assessment-investigation',
    title: 'The Algorithm That Taxes the Poor',
    summary:
      'A longform property tax investigation into regressivity, uniformity, and whether low-value homes are carrying more than their fair share.',
    tags: ['Property Tax', 'Equity', 'Longform'],
    featured: true,
    requiredFiles: [
      'index.html',
      'figures/01_regressivity_chart.png',
      'figures/02_quintile_bars.png',
      'figures/03_distribution.png',
      'figures/04_iaao_compliance.png',
      'figures/07_comprehensive_equity.png',
      'figures/08_historical_trends.png'
    ],
    publishFiles: [
      'index.html',
      'figures/01_regressivity_chart.png',
      'figures/02_quintile_bars.png',
      'figures/03_distribution.png',
      'figures/04_iaao_compliance.png',
      'figures/07_comprehensive_equity.png',
      'figures/08_historical_trends.png'
    ],
    buildHint: 'python explorations/property-assessment-investigation/property_assessment_investigation.py',
  },
  {
    slug: '311-service-equity-investigation',
    title: 'The 311 Gap',
    summary:
      'A civic equity investigation into whether response times stretch longer in lower-income neighborhoods, with tract-level demographic overlays.',
    tags: ['311', 'Demographics', 'Service Equity'],
    featured: false,
    requiredFiles: ['index.html', 'tract_data.json', 'census_tracts.geojson'],
    publishFiles: ['index.html', 'tract_data.json', 'census_tracts.geojson'],
    buildHint: 'Export tract_data.json and census_tracts.geojson before publishing this artifact.',
  },
  {
    slug: 'philly-transit',
    title: 'Philly Transit 3D',
    summary:
      'A pitched 3D SEPTA vehicle map with route-clamped animation, click-to-inspect details, and a 24-hour replay control.',
    tags: ['Transit', '3D', 'GTFS-RT'],
    featured: true,
    requiredFiles: ['index.html', 'help.html', 'styles.css', 'scripts', 'data/transit.json'],
    publishFiles: ['index.html', 'help.html', 'styles.css', 'scripts', 'data/transit.json'],
    buildHint: 'uv run python explorations/philly-transit/build_data.py',
  },
];
