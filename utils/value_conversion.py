"""Parse Football Manager transfer-value strings into floats."""
import re

import numpy as np

# A bare number with an optional magnitude suffix, e.g. "1.5M", "750K", "0".
_VALUE_RE = re.compile(r'^(?P<num>\d+(?:\.\d+)?)\s*(?P<suffix>[KMB])?$', re.IGNORECASE)
_MULTIPLIERS = {'K': 1e3, 'M': 1e6, 'B': 1e9}

# Currency symbols and separators FM emits across locales.
_STRIP_CHARS = str.maketrans('', '', '£$€,  ')

# Values that genuinely mean "no fee".
_FREE = {'0', 'FREE', 'FREETRANSFER'}
# Values that mean "unknown" -- distinct from free, so they become NaN.
_UNKNOWN = {'', 'N/A', 'NA', '-', '?'}


def value_to_float(raw):
    """Convert a transfer value to a float, or NaN if it cannot be parsed.

    Handles single values ("£1.5M") and ranges ("£1.2M - £1.8M", averaged).
    Returns NaN rather than 0.0 for unparseable input so that a parse failure
    is visibly missing instead of masquerading as a free player.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    if not isinstance(raw, str):
        return np.nan

    text = raw.translate(_STRIP_CHARS)
    upper = text.upper()
    if upper in _FREE:
        return 0.0
    if upper in _UNKNOWN:
        return np.nan

    # Split on a hyphen that separates two values, e.g. "1.2M-1.8M".
    parts = [p for p in text.split('-') if p]
    if len(parts) == 2:
        low, high = _parse_single(parts[0]), _parse_single(parts[1])
        if np.isnan(low) or np.isnan(high):
            return np.nan
        return (low + high) / 2
    if len(parts) == 1:
        return _parse_single(parts[0])
    return np.nan


def _parse_single(text):
    match = _VALUE_RE.match(text)
    if match is None:
        return np.nan
    number = float(match.group('num'))
    suffix = match.group('suffix')
    return number * _MULTIPLIERS[suffix.upper()] if suffix else number
