import type { Dataset } from "./catalog.ts";
// Stable identities and reviewed source scope, independent of HTTP implementation.
export const providerCatalog: Dataset[] = [
  {
    id: "loc:philadelphia_engineering",
    sourceId: "loc",
    title: "Philadelphia engineering records (Library of Congress)",
    description:
      "Historic American Buildings, Landscapes, and Engineering Records collection search for Philadelphia engineering. Search relevance order; paging only, no structured filters or custom ordering. Rights vary by item; consult each record.",
    organization: "Library of Congress",
    license: null,
    categories: ["History", "Engineering"],
    resources: [
      {
        id: "0000000000000001",
        name: "Philadelphia engineering search",
        format: "api",
        url: "https://www.loc.gov/collections/historic-american-buildings-landscapes-and-engineering-records/?q=philadelphia+engineering",
        query: null,
      },
    ],
  },
  {
    id: "usgs:schuylkill_latest",
    sourceId: "usgs",
    title: "Schuylkill River latest observations (USGS)",
    description:
      "Latest continuous observations for monitoring location USGS-01474500, Schuylkill River at Philadelphia. Values retain source units, timestamps and quality status. Paging only; not a historical series or flood forecast.",
    organization: "US Geological Survey",
    license: null,
    categories: ["Rivers", "Environment"],
    resources: [
      {
        id: "0000000000000002",
        name: "Schuylkill latest continuous observations",
        format: "api",
        url: "https://api.waterdata.usgs.gov/ogcapi/v1/collections/latest-continuous/items?monitoring_location_id=USGS-01474500",
        query: null,
      },
    ],
  },
  {
    id: "musicbrainz:philadelphia_places",
    sourceId: "musicbrainz",
    title: "Philadelphia music places (MusicBrainz)",
    description:
      "MusicBrainz place search where associated area matches Philadelphia; includes studios and closed venues, not only active music venues. Coordinates and founded/closed dates are optional MusicBrainz assertions, not verified city boundaries or current operation. Relevance order; paging only. One request per second per service instance; no retries.",
    organization: "MetaBrainz Foundation",
    license:
      "MusicBrainz core data CC0; see https://musicbrainz.org/doc/About/Data_License",
    categories: ["Music", "Places"],
    resources: [
      {
        id: "0000000000000003",
        name: "Philadelphia place search",
        format: "api",
        url: "https://musicbrainz.org/ws/2/place/?query=area%3A%22Philadelphia%22&fmt=json",
        query: null,
      },
    ],
  },
  {
    id: "noaa:philadelphia_tides_today",
    sourceId: "noaa",
    title: "Philadelphia today's hourly tide predictions (NOAA)",
    description:
      "Today's UTC hourly tide predictions for station 8545240, Philadelphia. Meters relative to MLLW; predictions, not observed water levels. The 24-hour scope is fixed; paging only.",
    organization: "NOAA CO-OPS",
    license: null,
    categories: ["Tides", "Rivers"],
    resources: [
      {
        id: "0000000000000004",
        name: "Philadelphia hourly predictions today UTC",
        format: "api",
        url: "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?station=8545240&product=predictions&date=today&interval=h&datum=MLLW&time_zone=gmt&units=metric&format=json",
        query: null,
      },
    ],
  },
];
export const providerByDataset = new Map(providerCatalog.map((d) => [d.id, d]));
