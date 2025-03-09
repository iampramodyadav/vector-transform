import pytest
import numpy as np
from unittest.mock import patch
from vector_transform.vector3d import (
    orderMult,
    coordinateTransform,
    fit_plane_pca,
    fit_plane_lsq,
    angle_between_planes,
    plane_intersection,
    data_density_plot,
)

# Tolerance for floating-point comparisons
TOL = 1e-6

# Fixtures for reusable test data
@pytest.fixture
def sample_point():
    return [1, 2, 3]

@pytest.fixture
def sample_points_3d():
    return np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])

# Tests for orderMult
def test_orderMult_typ1():
    Tr = np.eye(4)  # Identity matrix for translation
    Rx = np.eye(4)  # Identity matrix for rotation
    Ry = np.eye(4)
    Rz = np.eye(4)
    result = orderMult("TXYZ", Tr, Rx, Ry, Rz, typ=1)
    expected = Rz @ Ry @ Rx @ Tr  # Typ 1 multiplies in reverse order
    assert np.allclose(result, expected, atol=TOL)

def test_orderMult_typ2():
    Tr = np.eye(4)
    Rx = np.eye(4)
    Ry = np.eye(4)
    Rz = np.eye(4)
    result = orderMult("TXYZ", Tr, Rx, Ry, Rz, typ=2)
    expected = Tr @ Rx @ Ry @ Rz  # Typ 2 multiplies in forward order
    assert np.allclose(result, expected, atol=TOL)

# Tests for coordinateTransform
def test_coordinateTransform_typ1_identity(sample_point):
    result = coordinateTransform(order="TXYZ", Tra=[0, 0, 0], a_x=0, a_y=0, a_z=0, point=sample_point, typ=1)
    assert np.allclose(result, sample_point, atol=TOL), "Identity transform should return original point"

def test_coordinateTransform_typ2_translation(sample_point):
    result = coordinateTransform(order="TXYZ", Tra=[1, 1, 1], a_x=0, a_y=0, a_z=0, point=sample_point, typ=2)
    expected = [2, 3, 4]  # Point moves by translation
    assert np.allclose(result, expected, atol=TOL)

# Tests for fit_plane_pca
def test_fit_plane_pca_collinear(sample_points_3d):
    normal, d = fit_plane_pca(sample_points_3d)
    assert np.allclose(np.linalg.norm(normal), 1, atol=TOL), "Normal vector should be unit length"
    # Check that all points lie on the plane (or close due to numerical precision)
    for point in sample_points_3d:
        assert abs(np.dot(normal, point) + d) < TOL

# Tests for fit_plane_lsq
def test_fit_plane_lsq_collinear(sample_points_3d):
    normal, d = fit_plane_lsq(sample_points_3d)
    assert np.allclose(np.linalg.norm(normal), np.sqrt(normal[0]**2 + normal[1]**2 + 1), atol=TOL)
    # Check plane equation holds approximately
    for point in sample_points_3d:
        assert abs(np.dot(normal, point) + d) < TOL * 10  # Relaxed tolerance due to least squares

# Tests for angle_between_planes
def test_angle_between_planes_parallel():
    normal1 = np.array([0, 0, 1])
    normal2 = np.array([0, 0, 2])
    angle = angle_between_planes(normal1, normal2)
    assert np.allclose(angle, 0, atol=TOL), "Parallel planes should have 0 angle"

def test_angle_between_planes_perpendicular():
    normal1 = np.array([1, 0, 0])
    normal2 = np.array([0, 1, 0])
    angle = angle_between_planes(normal1, normal2)
    assert np.allclose(angle, np.pi / 2, atol=TOL), "Perpendicular planes should have 90-degree angle"

# Tests for plane_intersection
def test_plane_intersection_parallel():
    normal1 = np.array([0, 0, 1])
    d1 = 1
    normal2 = np.array([0, 0, 2])
    d2 = 2
    point, direction = plane_intersection(normal1, d1, normal2, d2)
    assert np.all(np.isinf(direction)), "Parallel planes should have infinite direction vector"

def test_plane_intersection_perpendicular():
    normal1 = np.array([1, 0, 0])
    d1 = 0
    normal2 = np.array([0, 1, 0])
    d2 = 0
    point, direction = plane_intersection(normal1, d1, normal2, d2)
    assert np.allclose(direction, [0, 0, 1], atol=TOL), "Intersection line should be along z-axis"
    assert np.allclose(point, [0, 0, 0], atol=TOL), "Intersection point should include origin"

# Tests for data_density_plot
@patch("matplotlib.pyplot.savefig")  # Mock savefig to avoid file creation
def test_data_density_plot(mock_savefig):
    data = np.random.normal(0, 1, 100)
    result = data_density_plot(data, "Test Title", "Test Subtitle", unit="m")
    assert "PDF" in result and "Normal" in result, "Result should contain PDF and Normal keys"
    assert len(result["Normal"]) == 2, "Normal should contain mu and sigma"
    assert isinstance(result["Normal"][0], float) and isinstance(result["Normal"][1], float)
    mock_savefig.assert_called_once()

# Edge case tests
def test_coordinateTransform_invalid_typ(sample_point):
    with pytest.raises(SystemExit):  # Assuming print and exit behavior for invalid typ
        coordinateTransform(typ=3, point=sample_point)

def test_fit_plane_pca_empty():
    with pytest.raises(ValueError):
        fit_plane_pca(np.array([]))

if __name__ == "__main__":
    pytest.main(["-v"])
