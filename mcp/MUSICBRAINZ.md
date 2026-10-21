# Connecting MusicBrainz places to local Philadelphia records

`musicbrainz:philadelphia_places` remains one bounded place-search projection with the fixed query `area:"Philadelphia"`. The associated-area search is not a city-polygon check. Results may include studios, closed places and incomplete records. This adapter does not fetch recording relationships or establish where a recording was made.

## Fields and provenance

The existing `id`, `name`, `type`, `address` and `area` columns remain. Optional source fields are exposed without adding lookup requests:

| Column | Type | Source and meaning |
| --- | --- | --- |
| `latitude`, `longitude` | number or null, independently nullable | `coordinates.latitude` / `.longitude`, converted from validated decimal strings; WGS84 degrees within ±90 / ±180 |
| `begin_date`, `end_date` | string or null | `life-span.begin` / `.end`, founded/closed dates with source precision preserved, such as `1971` or `1999-10` |
| `ended` | boolean or null | `life-span.ended` as supplied; false is preserved, absence is unknown, and neither date nor flag is treated as verified current status |
| `place_url` | string | Canonical `https://musicbrainz.org/place/{id}`, built from the validated MBID; this URL is returned for citation and never fetched |

Missing or explicitly null coordinate/lifespan objects and members yield nulls. Empty date strings yield null; dates are not padded to January 1 or converted into timestamps. Invalid present coordinate values or boolean types produce the existing structured upstream error instead of a fabricated usable value. `get_schema` identifies the normalized projection and each added field's source path. The result envelope retains source/resource IDs, the exact search URL, retrieval timestamp, license and completeness metadata.

Contract references: [official place-search JSON example](https://musicbrainz.org/doc/MusicBrainz_API/Search#Place) and [place properties](https://musicbrainz.org/doc/Place). The checked-in `test/fixtures/musicbrainz-place-search.json` transcribes the documentation's Chipping Norton example to verify field names and types; it is explicitly not a Philadelphia/live-response fixture. Additional tests use synthetic null/invalid cases. Previous upstream probes were denied by the environment CONNECT proxy (403), and this extension makes no further live request or coverage claim.

## Address candidates with location context

Given a local CSV with `local_id` and `address`, this SDK example produces address-match candidates for one bounded page. It performs no external geocoding or additional provider calls. Run against a server containing this draft adapter; the public endpoint changes only after a coordinated deployment.

```python
import asyncio
import unicodedata
import pandas as pd
from philly import RemotePhilly


def address_key(value):
    if not isinstance(value, str) or not value.strip():
        return None
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split())


async def main():
    remote = RemotePhilly("http://127.0.0.1:8787/mcp")
    page = await remote.query("musicbrainz:philadelphia_places", limit=20)
    music = pd.DataFrame(page["rows"], columns=page["columns"])
    local = pd.read_csv("local_places.csv", dtype={"address": "string"})
    music["address_key"] = music["address"].map(address_key)
    local["address_key"] = local["address"].map(address_key)
    candidates = local.dropna(subset=["address_key"]).merge(
        music.dropna(subset=["address_key"]),
        on="address_key", how="inner", suffixes=("_local", "_musicbrainz"),
    )
    # Duplicate addresses remain multiple candidates. MBID is the place key.
    for key in ("sourceId", "sourceUrl", "retrievedAt"):
        candidates[key] = page[key]
    print(candidates)  # includes record URL, coordinates and partial dates
    print({key: page[key] for key in (
        "truncated", "completeness", "completenessReasons", "nextOffset"
    )})


asyncio.run(main())
```

Address equality is a candidate link, not proof of the same venue: spelling, units, tenants and historical reuse can differ. Review names, record links, partial dates and coordinate availability before accepting a match. If a trusted local layer also has coordinates, compare both in a suitable metric CRS (for example EPSG:26918 near Philadelphia) to review location proximity; a nearby point alone still does not establish identity or a recording relationship. Missing coordinates stay missing rather than being inferred from the area name. Retain the original page envelope alongside any exported candidates; paging/index changes and truncation limit coverage.

CLI consumers receive the same JSON fields:

```sh
phl remote schema musicbrainz:philadelphia_places --endpoint http://127.0.0.1:8787/mcp
phl remote query musicbrainz:philadelphia_places --limit 20 --endpoint http://127.0.0.1:8787/mcp
```

Selected columns, native paging, the one-second service-instance cooldown, result bounds and explicit rejection of provider filters/order/aggregation remain unchanged.
