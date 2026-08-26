"""Procedural geometry: platonic solids, icosphere subdivision, normals."""
from __future__ import annotations

import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def icosahedron() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([
        [-1, PHI, 0], [1, PHI, 0], [-1, -PHI, 0], [1, -PHI, 0],
        [0, -1, PHI], [0, 1, PHI], [0, -1, -PHI], [0, 1, -PHI],
        [PHI, 0, -1], [PHI, 0, 1], [-PHI, 0, -1], [-PHI, 0, 1],
    ], dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)
    return v, f


def tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    return v, f


def cube() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)],
                 float) / np.sqrt(3)
    f = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return v, f


def octahedron() -> tuple[np.ndarray, np.ndarray]:
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                  [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    f = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
                  [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    return v, f


def subdivide(verts: np.ndarray, faces: np.ndarray,
              times: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Loop-style midpoint subdivision, vertices re-projected onto unit sphere."""
    for _ in range(times):
        verts = list(map(np.asarray, verts))
        cache: dict[tuple[int, int], int] = {}
        new_faces = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            m = (verts[a] + verts[b]) / 2
            m = m / np.linalg.norm(m)
            verts.append(m)
            cache[key] = len(verts) - 1
            return cache[key]

        for tri in faces:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.asarray(verts)
        faces = np.asarray(new_faces, dtype=np.int64)
    return np.asarray(verts), faces


def icosphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    v, f = icosahedron()
    return subdivide(v, f, subdivisions)


def face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    n = np.cross(b - a, c - a)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def vertex_normals_sphere(verts: np.ndarray) -> np.ndarray:
    """For sphere-like meshes, normals are just normalized positions."""
    return verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-12)


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])
