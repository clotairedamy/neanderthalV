import numpy as np

from visualizer.physics.velocity import VelocityArray, VelocityValue, VelocityVector
from visualizer.viz import geometry as G
from visualizer.viz.mono import morph_sphere


def test_velocity_value_converges_and_impulses():
    v = VelocityValue(0, accel=10, damping=0.85)
    v.set_target(1.0)
    for _ in range(240):
        v.update(1 / 60)
    assert abs(v.value - 1.0) < 0.1
    v.impulse(5.0)
    v.update(1 / 60)
    assert v.speed > 1.0


def test_velocity_vector_and_array():
    vec = VelocityVector([0, 0, 0])
    vec.set_target([1, 2, 3])
    for _ in range(240):
        vec.update(1 / 60)
    assert np.allclose(vec.value, [1, 2, 3], rtol=0.1)
    arr = VelocityArray(7)
    arr.set_target(np.ones(7))
    for _ in range(240):
        arr.update(1 / 60)
    assert np.all(np.abs(arr.value - 1) < 0.2)


def test_icosphere_subdivision():
    for sub in range(4):
        verts, faces = G.icosphere(sub)
        assert faces.max() < len(verts)
        assert np.allclose(np.linalg.norm(verts, axis=1), 1.0, atol=1e-6)
    assert len(G.icosphere(3)[0]) == 642


def test_platonic_solids_valid():
    for fn in (G.tetrahedron, G.cube, G.octahedron, G.icosahedron):
        v, f = fn()
        assert f.max() < len(v)


def test_rotation_matrix():
    R = G.rotation_matrix(np.array([0, 1, 0]), np.pi / 2)
    assert np.allclose(R @ np.array([1, 0, 0]), [0, 0, -1], atol=1e-9)


def test_morph_sphere_bounded_and_identity():
    v, _ = G.icosphere(2)
    out = morph_sphere(v, t=1.0, amount=0.0, twist=0.0)
    assert np.allclose(out, v, atol=1e-6)
    out = morph_sphere(v, t=2.0, amount=0.3, twist=0.8)
    r = np.linalg.norm(out, axis=1)
    assert r.min() > 0.5 and r.max() < 1.5
