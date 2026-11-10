"""Triangle-mesh utilities (pure numpy): edge subdivision templates, splitting a mesh
along the zero set of a scalar field (used to cut exact material boundaries into the
car skin), local refinement, misc helpers."""
import numpy as np


def unique_edges(T, n_verts):
    a = T[:, [0, 1, 2]].ravel()
    b = T[:, [1, 2, 0]].ravel()
    lo = np.minimum(a, b).astype(np.int64)
    hi = np.maximum(a, b).astype(np.int64)
    key = np.unique(lo * n_verts + hi)
    return key // n_verts, key % n_verts


def subdivide(V, T, lo, hi, t):
    """Insert a vertex on each undirected edge (lo[i] < hi[i]) at parameter t[i] from lo.
    Returns (V2, T2, parent) where parent[j] is the source triangle of new triangle j."""
    N = len(V)
    if len(lo) == 0:
        return V, T, np.arange(len(T))
    t = np.broadcast_to(np.asarray(t, dtype=np.float64), lo.shape)
    newp = V[lo] + (V[hi] - V[lo]) * t[:, None]
    V2 = np.vstack([V, newp])
    key = lo.astype(np.int64) * N + hi.astype(np.int64)
    order = np.argsort(key)
    skey = key[order]

    def lookup(a, b):
        l = np.minimum(a, b).astype(np.int64)
        h = np.maximum(a, b).astype(np.int64)
        k = l * N + h
        pos = np.clip(np.searchsorted(skey, k), 0, len(skey) - 1)
        found = skey[pos] == k
        return np.where(found, N + order[pos], -1)

    v0, v1, v2 = T[:, 0], T[:, 1], T[:, 2]
    m0 = lookup(v0, v1)
    m1 = lookup(v1, v2)
    m2 = lookup(v2, v0)
    code = (m0 >= 0).astype(np.int8) + 2 * (m1 >= 0) + 4 * (m2 >= 0)
    tris = []
    parents = []
    idx = np.arange(len(T))

    def emit(mask, *cols):
        if not mask.any():
            return
        for c in cols:
            tris.append(np.stack([c[0][mask], c[1][mask], c[2][mask]], axis=1))
            parents.append(idx[mask])

    emit(code == 0, (v0, v1, v2))
    emit(code == 1, (v0, m0, v2), (m0, v1, v2))
    emit(code == 2, (v1, m1, v0), (m1, v2, v0))
    emit(code == 4, (v2, m2, v1), (m2, v0, v1))
    emit(code == 3, (m0, v1, m1), (v0, m0, m1), (v0, m1, v2))
    emit(code == 6, (m1, v2, m2), (v1, m1, m2), (v1, m2, v0))
    emit(code == 5, (m2, v0, m0), (v2, m2, m0), (v2, m0, v1))
    emit(code == 7, (v0, m0, m2), (m0, v1, m1), (m2, m1, v2), (m0, m1, m2))
    T2 = np.concatenate(tris, axis=0).astype(np.int32)
    parent = np.concatenate(parents, axis=0)
    return V2, T2, parent


def split_by_field(V, T, d, snap=0.22, tri_mask=None, locked=None):
    """Cut triangles along d == 0 (d given per vertex, linear along edges).
    Vertices close to the cut are snapped onto it instead of creating slivers.
    tri_mask (optional, per triangle bool) restricts which triangles may be cut.
    Returns (V2, T2, parent, touched) - touched marks vertices that moved or are new."""
    V = V.copy()
    d = d.astype(np.float64).copy()
    N = len(V)
    Tsel = T if tri_mask is None else T[tri_mask]
    touched = np.zeros(N, dtype=bool)
    if len(Tsel) == 0:
        return V, T, np.arange(len(T)), touched
    lo, hi = unique_edges(Tsel, N)
    cross = d[lo] * d[hi] < 0
    if not cross.any():
        return V, T, np.arange(len(T)), touched
    lo_c, hi_c = lo[cross], hi[cross]
    t = d[lo_c] / (d[lo_c] - d[hi_c])
    # --- snap pass
    if snap > 0:
        P = V[lo_c] + (V[hi_c] - V[lo_c]) * t[:, None]
        s_lo = t < snap
        s_hi = t > 1.0 - snap
        if locked is not None:          # vertices sitting on earlier cut lines must not move
            s_lo &= ~locked[lo_c]
            s_hi &= ~locked[hi_c]
        acc = np.zeros((N, 3)); cnt = np.zeros(N)
        np.add.at(acc, lo_c[s_lo], P[s_lo]); np.add.at(cnt, lo_c[s_lo], 1)
        np.add.at(acc, hi_c[s_hi], P[s_hi]); np.add.at(cnt, hi_c[s_hi], 1)
        m = cnt > 0
        V[m] = acc[m] / cnt[m][:, None]
        d[m] = 0.0
        touched |= m
        cross = d[lo] * d[hi] < 0
        lo_c, hi_c = lo[cross], hi[cross]
        t = d[lo_c] / (d[lo_c] - d[hi_c])
    V2, T2, parent = subdivide(V, T, lo_c, hi_c, t)
    touched = np.concatenate([touched, np.ones(len(V2) - N, dtype=bool)])
    return V2, T2, parent, touched


def refine_long_edges(V, T, edge_sel_fn, max_len, project_fn=None, iters=5):
    """Bisect edges longer than max_len for which edge_sel_fn(midpoints) is True."""
    parent = np.arange(len(T))
    for _ in range(iters):
        N = len(V)
        lo, hi = unique_edges(T, N)
        L = np.linalg.norm(V[hi] - V[lo], axis=1)
        cand = L > max_len
        if not cand.any():
            break
        mid = 0.5 * (V[lo[cand]] + V[hi[cand]])
        sel = np.zeros(len(lo), dtype=bool)
        sel[np.nonzero(cand)[0]] = edge_sel_fn(mid)
        if not sel.any():
            break
        V, T, p = subdivide(V, T, lo[sel], hi[sel], 0.5)
        parent = parent[p]
        if project_fn is not None:
            V[N:] = project_fn(V[N:])
    return V, T, parent


def compact(V, T, *vattrs):
    """Drop unreferenced vertices. Returns (V2, T2, [attrs...])."""
    used = np.zeros(len(V), dtype=bool)
    used[T.ravel()] = True
    remap = np.full(len(V), -1, dtype=np.int64)
    remap[used] = np.arange(int(used.sum()))
    out = [V[used], remap[T].astype(np.int32)]
    for a in vattrs:
        out.append(a[used])
    return out


def tri_normals(V, T):
    n = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    return n


def area_vertex_normals(V, T):
    fn = tri_normals(V, T)
    out = np.zeros_like(V)
    for k in range(3):
        np.add.at(out, T[:, k], fn)
    l = np.linalg.norm(out, axis=1)
    return out / np.maximum(l, 1e-12)[:, None]


def auto_smooth_split(V, T, angle_deg=40.0):
    """Split vertices across edges sharper than angle so that area-weighted normals give
    crisp creases. Returns (V2, T2, normals)."""
    fn = tri_normals(V, T)
    fl = np.linalg.norm(fn, axis=1)
    fnu = fn / np.maximum(fl, 1e-20)[:, None]
    cosang = np.cos(np.radians(angle_deg))
    # corner -> smoothing group via union-find over faces around each vertex
    nt = len(T)
    corner_vert = T.ravel()
    order = np.argsort(corner_vert, kind='stable')
    sorted_v = corner_vert[order]
    starts = np.flatnonzero(np.r_[True, sorted_v[1:] != sorted_v[:-1]])
    ends = np.r_[starts[1:], len(sorted_v)]
    newV = []
    newN = []
    T2 = T.copy().ravel()
    next_id = 0
    for s, e in zip(starts, ends):
        corners = order[s:e]
        faces = corners // 3
        normals = fnu[faces]
        k = len(faces)
        group = -np.ones(k, dtype=np.int64)
        g = 0
        for i in range(k):
            if group[i] >= 0:
                continue
            group[i] = g
            stack = [i]
            while stack:
                a = stack.pop()
                close = (normals @ normals[a] > cosang) & (group < 0)
                idxs = np.flatnonzero(close)
                group[idxs] = g
                stack.extend(idxs.tolist())
            g += 1
        v = sorted_v[s]
        for gi in range(g):
            m = group == gi
            nsum = fn[faces[m]].sum(axis=0)
            nsum /= max(np.linalg.norm(nsum), 1e-20)
            newV.append(V[v]); newN.append(nsum)
            T2[corners[m]] = next_id
            next_id += 1
    return np.array(newV), T2.reshape(-1, 3).astype(np.int32), np.array(newN)


def merge_meshes(parts):
    """parts: list of dicts with V, T, N (normals), M (per-tri material id), optional C (per-vertex ao).
    Returns a single dict."""
    Vs, Ts, Ns, Ms, Cs = [], [], [], [], []
    off = 0
    for p in parts:
        if len(p['T']) == 0:
            continue
        Vs.append(p['V']); Ns.append(p['N']); Ts.append(p['T'] + off); Ms.append(p['M'])
        Cs.append(p['C'] if p.get('C') is not None else np.ones(len(p['V'])))
        off += len(p['V'])
    return dict(V=np.vstack(Vs), T=np.vstack(Ts).astype(np.int32), N=np.vstack(Ns),
                M=np.concatenate(Ms).astype(np.int32), C=np.concatenate(Cs))


def lathe(profile, segments, axis='x', close_profile=False):
    """Revolve a 2D profile [(a, r), ...] (a = position along the axis, r = radius) around `axis`.
    Returns (V, T, N). Normals are computed from the profile (smooth around, smooth along unless
    consecutive duplicate points are used to create creases)."""
    prof = np.asarray(profile, dtype=np.float64)
    m = len(prof)
    # profile normals (2D): rotate tangent; duplicated points -> crease (each copy takes its own segment normal)
    tang = np.zeros_like(prof)
    for i in range(m):
        p_prev = prof[i - 1] if i > 0 else None
        p_next = prof[i + 1] if i < m - 1 else None
        tv = np.zeros(2)
        if p_prev is not None and np.linalg.norm(prof[i] - p_prev) > 1e-9:
            d = prof[i] - p_prev; tv += d / np.linalg.norm(d)
        if p_next is not None and np.linalg.norm(p_next - prof[i]) > 1e-9:
            d = p_next - prof[i]; tv += d / np.linalg.norm(d)
        if np.linalg.norm(tv) < 1e-9:
            tv = np.array([1.0, 0.0])
        tang[i] = tv / np.linalg.norm(tv)
    # outward normal for a profile running in +a direction with r>0 is (-dr, da) -> (n_a, n_r)
    na = -tang[:, 1]
    nr = tang[:, 0]
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    V = np.zeros((segments, m, 3)); N = np.zeros((segments, m, 3))
    V[:, :, 0] = prof[None, :, 0]
    V[:, :, 1] = prof[None, :, 1] * ca[:, None]
    V[:, :, 2] = prof[None, :, 1] * sa[:, None]
    N[:, :, 0] = na[None, :]
    N[:, :, 1] = nr[None, :] * ca[:, None]
    N[:, :, 2] = nr[None, :] * sa[:, None]
    V = V.reshape(-1, 3); N = N.reshape(-1, 3)
    T = []
    for s in range(segments):
        s2 = (s + 1) % segments
        for i in range(m - 1):
            if np.linalg.norm(prof[i + 1] - prof[i]) < 1e-9:
                continue
            a = s * m + i; b = s * m + i + 1; c = s2 * m + i + 1; d = s2 * m + i
            if prof[i, 1] < 1e-9:      # degenerate at the axis
                T.append((a, b, c))
            elif prof[i + 1, 1] < 1e-9:
                T.append((a, b, d))
            else:
                T.append((a, b, c)); T.append((a, c, d))
    T = np.array(T, dtype=np.int32)
    # orientation check: make normals agree with geometric normals
    gn = tri_normals(V, T)
    agree = (gn * (N[T[:, 0]] + N[T[:, 1]] + N[T[:, 2]])).sum(1)
    flip = agree < 0
    T[flip] = T[flip][:, ::-1]
    if axis == 'y':
        V = V[:, [1, 0, 2]] * np.array([1, 1, -1.0]); N = N[:, [1, 0, 2]] * np.array([1, 1, -1.0])
    elif axis == 'z':
        V = V[:, [2, 1, 0]] * np.array([-1.0, 1, 1]); N = N[:, [2, 1, 0]] * np.array([-1.0, 1, 1])
    return V, T, N
