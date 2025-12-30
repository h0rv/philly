from urllib.parse import quote


def normalize_url(url: str) -> str:
    """Normalize and encode URLs while preserving common URL separators."""
    collapsed = " ".join(url.split())
    return quote(collapsed, safe=":/?&=%")
