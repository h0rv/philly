# Philly

<img src="./assets/logo.png" width="256" alt="logo">

Python Swiff Army knife and CLI for working with [OpenDataPhilly](https://opendataphilly.org/) datasets.

## Examples

```python
from philly import Philly

ps = Philly()

# TODO:
print(ps.list_datasets())
```

### CLI

#### Install

Install globally:

```
uv tool install philly
```

Run directly:

```bash
uvx --from philly phl
```

Help/Usage:

```bash
phl
```

```bash
phl list-datasets
```

Search datasets with fuzzy search ([`fzf` install instructions](https://github.com/junegunn/fzf?tab=readme-ov-file#installation))

```bash
phl list-datasets | fzf
```

```bash
phl list-all-resources | fzf
```

Interactively find and load a resource:

```bash
phl list-datasets \
    | fzf \
    | xargs -I {} bash -c \
    'phl list-resources "{}" --names-only | fzf | xargs -I @ phl load "{}" "@"'
```

## Development

### Update Datasets

```bash
uv run scripts/update_datasets.py
```

### CLI

#### Install CLI

```bash
uv tool install phl --editable
```

## Resources

* OpenDataPhilly
    * https://opendataphilly.org/
    * https://github.com/opendataphilly/
