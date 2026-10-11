import pytest
from src.geospatial.grid import cell_centers


def test_count_and_spacing():
    g = cell_centers(0, 0, 1000, 500, 250)
    assert len(g) == 8 and g.cell_id.is_unique
    assert sorted(g.x.unique()) == [125, 375, 625, 875]


def test_grid_is_deterministic_and_snapped():
    a = cell_centers(1010, 2020, 1600, 2600, 250)
    b = cell_centers(1010, 2020, 1600, 2600, 250)
    assert a.equals(b)
    assert (a.x - 125) .mod(250).eq(0).all()      # centres sit on multiples of size + half


def test_bad_inputs():
    with pytest.raises(ValueError):
        cell_centers(0, 0, 100, 100, 0)
    with pytest.raises(ValueError):
        cell_centers(100, 0, 0, 100, 10)
