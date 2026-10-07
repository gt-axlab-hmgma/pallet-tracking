from palletvision_zones.geometry import ground_point


def test_ground_point_is_bottom_centre_of_box() -> None:
    assert ground_point((100, 50, 200, 150)) == (150.0, 150.0)


def test_ground_point_accepts_floats() -> None:
    assert ground_point((0.5, 1.0, 1.5, 3.0)) == (1.0, 3.0)
