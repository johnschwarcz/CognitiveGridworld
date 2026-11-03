"""Shrink-wrapping a point cloud and counting its holes, with numpy and scipy only (28 Oct 2026; for the figure-8
question: does the wrapped surface of the Optimal's states have two holes?).

topology(B)      the Betti numbers of a voxel set B (3-D bool array, every True voxel a closed unit cube): b0 pieces
                 (26-connected), b2 enclosed cavities (6-connected components of the complement, minus the outside),
                 the Euler characteristic chi = V - E + F - C of the cube complex, b1 = b0 + b2 - chi: the holes
                 (tunnels). For one piece without cavities, b1 is the genus of its wrapping surface (2: a double torus,
                 the thickened figure 8)
density(P, ...)  the points P (n, 3) binned on a voxel grid and Gaussian-smoothed
march(F, level)  the iso-surface F = level by marching tetrahedra (each grid cube cut into 6 tetrahedra around its
                 main diagonal; 1 or 2 triangles per tetrahedron), vertices in grid units -> (m, 3, 3) triangles
check()          the counter on shapes of known genus (a cube, a solid torus, a hollow shell, a double torus)"""
import numpy as np
from scipy import ndimage

_S26 = np.ones((3, 3, 3), bool)
_S6 = ndimage.generate_binary_structure(3, 1)


def topology(B):
    """-> dict(b0, b1, b2, chi) of the union of the closed unit cubes of B"""
    B = np.asarray(B, bool)
    P = np.pad(B, 1)
    C = int(B.sum())
    F = sum(int((P[s0] | P[s1]).sum()) for s0, s1 in (
        ((slice(None, -1), slice(1, -1), slice(1, -1)), (slice(1, None), slice(1, -1), slice(1, -1))),
        ((slice(1, -1), slice(None, -1), slice(1, -1)), (slice(1, -1), slice(1, None), slice(1, -1))),
        ((slice(1, -1), slice(1, -1), slice(None, -1)), (slice(1, -1), slice(1, -1), slice(1, None)))))
    E = 0
    for ax in range(3):                                                           # edges along axis ax
        o = [a for a in range(3) if a != ax]
        acc = np.zeros(1, bool)
        for d0 in (0, 1):
            for d1 in (0, 1):
                sl = [slice(1, -1)] * 3
                sl[o[0]] = slice(d0, P.shape[o[0]] - 1 + d0)
                sl[o[1]] = slice(d1, P.shape[o[1]] - 1 + d1)
                acc = acc | P[tuple(sl)]
        E += int(acc.sum())
    V_ = np.zeros(1, bool)
    for d in np.ndindex(2, 2, 2):
        V_ = V_ | P[tuple(slice(k, P.shape[a] - 1 + k) for a, k in enumerate(d))]
    V = int(V_.sum())
    chi = V - E + F - C
    b0 = ndimage.label(B, _S26)[1]
    b2 = ndimage.label(~P, _S6)[1] - 1
    return dict(b0=b0, b1=b0 + b2 - chi, b2=b2, chi=chi)


def density(P, lo, hi, step, sigma):
    """points (n, 3) -> the counts on the grid [lo, hi) of voxel size step, Gaussian-smoothed (sigma voxels), and the
    grid's voxel centres along each axis"""
    n = np.ceil((np.asarray(hi) - lo) / step).astype(int)
    idx = np.floor((P - lo) / step).astype(int)
    ok = ((idx >= 0) & (idx < n)).all(1)
    flat = np.ravel_multi_index(idx[ok].T, n)
    D = np.bincount(flat, minlength=int(np.prod(n))).reshape(n).astype(np.float64)
    return ndimage.gaussian_filter(D, sigma), [lo[a] + step * (np.arange(n[a]) + .5) for a in range(3)]


_CUBE = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
_TETS = np.array([[0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6]])


def march(F, level):
    """the iso-surface of the grid values F at level -> triangles (m, 3, 3) in grid coordinates"""
    nx, ny, nz = F.shape
    base = np.stack(np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij"), -1)
    base = base.reshape(-1, 3)
    tris = []
    for tet in _TETS:
        Pv = base[:, None, :] + _CUBE[tet][None]                                   # (n, 4, 3) corner positions
        Vv = F[Pv[..., 0], Pv[..., 1], Pv[..., 2]]                                  # (n, 4)
        ins = Vv > level
        k = ins.sum(1)

        def cut(i, j, m):                                                          # crossing points on edge i-j
            a, b = Vv[m, i], Vv[m, j]
            w = ((level - a) / np.where(b != a, b - a, 1))[:, None]
            return Pv[m, i] + w * (Pv[m, j] - Pv[m, i])

        for kk in (1, 3):                                                          # one lone vertex: a triangle
            m = np.flatnonzero(k == kk)
            if not len(m):
                continue
            lone = np.argmax(ins[m] if kk == 1 else ~ins[m], 1)
            for v in range(4):
                mm = m[lone == v]
                if len(mm):
                    others = [u for u in range(4) if u != v]
                    tris.append(np.stack([cut(v, u, mm) for u in others], 1))
        m = np.flatnonzero(k == 2)                                                 # two in, two out: a quad
        for pair in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            sel = ins[m][:, list(pair)].all(1)
            mm = m[sel]
            if not len(mm):
                continue
            a, b = pair
            c, d = [u for u in range(4) if u not in pair]
            p1, p2, p3, p4 = cut(a, c, mm), cut(a, d, mm), cut(b, d, mm), cut(b, c, mm)
            tris.append(np.stack([p1, p2, p3], 1))
            tris.append(np.stack([p1, p3, p4], 1))
    return np.concatenate(tris) if tris else np.zeros((0, 3, 3))


def check():
    g = np.stack(np.meshgrid(*[np.arange(-20, 21)] * 3, indexing="ij"), -1).astype(float)
    x, y, z = g[..., 0], g[..., 1], g[..., 2]
    cube = (np.abs(g) <= 8).all(-1)
    torus = (np.hypot(x, y) - 12) ** 2 + z ** 2 <= 4 ** 2
    shell = (np.linalg.norm(g, axis=-1) <= 15) & (np.linalg.norm(g, axis=-1) >= 11)
    t1 = (np.hypot(x - 8, y) - 6) ** 2 + z ** 2 <= 2.5 ** 2
    t2 = (np.hypot(x + 8, y) - 6) ** 2 + z ** 2 <= 2.5 ** 2
    for name, B, want in (("cube", cube, (1, 0, 0)), ("torus", torus, (1, 1, 0)), ("shell", shell, (1, 0, 1)),
                          ("double torus", t1 | t2, (1, 2, 0))):
        t = topology(B)
        got = (t["b0"], t["b1"], t["b2"])
        print(f"{name:13s} b0, b1, b2 = {got} (expected {want}) {'ok' if got == want else 'WRONG'}")


if __name__ == "__main__":
    check()
