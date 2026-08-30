import numpy as np
import pytest

from utils.value_conversion import value_to_float


@pytest.mark.parametrize('raw, expected', [
    ('£1.5M', 1_500_000),
    ('750K', 750_000),
    ('£40,000', 40_000),
    ('1.2M - 1.8M', 1_500_000),
    ('£1.2M - £1.8M', 1_500_000),
    ('0', 0.0),
    ('Free', 0.0),
    (1500.0, 1500.0),
    (0, 0.0),
])
def test_parses_known_forms(raw, expected):
    assert value_to_float(raw) == pytest.approx(expected)


@pytest.mark.parametrize('raw', ['garbage', 'N/A', '', '-', None, '£M', object()])
def test_unparseable_becomes_nan_not_zero(raw):
    """A parse failure must not masquerade as a free transfer."""
    assert np.isnan(value_to_float(raw))
