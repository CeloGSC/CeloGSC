"""Generate the profile banner.

Left: a LiDAR point cloud of a small plant. A vertical scan line sweeps left to right and
colors the segmented equipment.
Right: a voxelized breast DCE-MRI volume, cut open to show the tissue inside. A horizontal
scan line sweeps top to bottom, slice by slice, and marks the lesion.
The two sweeps run on different clocks, like two separate models.

Everything is synthetic (geometric shapes + seeded noise). Static SVG with SMIL animation,
so it needs no GitHub Actions or external service.
Usage: python make_banner.py OUT.svg
"""
import math
import sys

import numpy as np

rng = np.random.default_rng(7)
W, H = 1000, 320
SPLIT = 630  # x where the LiDAR panel ends and the MRI panel begins
GRAY = "#8b949e"
COLORS = {"tank": "#22a699", "pipe": "#e8962e", "pump": "#8a6cff", "structure": "#4f8ff7", "lesion": "#f2545b"}


# ---------- primitives (world units, z up) ----------
def jitter(p, s=0.025):
    return p + rng.normal(0, s, p.shape)


def cyl_v(cx, cy, r, z0, z1, d):
    n = int(2 * math.pi * r * (z1 - z0) * d)
    t = rng.uniform(0, 2 * math.pi, n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t), rng.uniform(z0, z1, n)]


def disk(cx, cy, z, r, d):
    n = int(math.pi * r * r * d)
    rr, t = r * np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * math.pi, n)
    return np.c_[cx + rr * np.cos(t), cy + rr * np.sin(t), np.full(n, z)]


def pipe(a, b, r, d):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    L = np.linalg.norm(ax)
    ax /= L
    tmp = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(ax, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    n = max(int(2 * math.pi * r * L * d), 8)
    t, s = rng.uniform(0, 2 * math.pi, n), rng.uniform(0, L, n)
    return a + np.outer(s, ax) + r * (np.outer(np.cos(t), e1) + np.outer(np.sin(t), e2))


def box(c, size, d):
    c, h = np.asarray(c, float), np.asarray(size, float) / 2
    out = []
    for axis in range(3):
        o = [i for i in range(3) if i != axis]
        n = int(size[o[0]] * size[o[1]] * d)
        for sign in (-1, 1):
            p = np.zeros((n, 3))
            p[:, axis] = sign * h[axis]
            p[:, o[0]] = rng.uniform(-h[o[0]], h[o[0]], n)
            p[:, o[1]] = rng.uniform(-h[o[1]], h[o[1]], n)
            out.append(p + c)
    return np.vstack(out)


# ---------- scene A: LiDAR scan of a small plant ----------
def plant():
    tank_c, tank_r = (5.5, 0.2), 2.0
    pump_c = (-4.0, 2.5)
    parts = {k: [] for k in ("ground", "tank", "pipe", "pump", "structure")}

    # ground: concentric scan rings around the scanner, like a terrestrial LiDAR
    scanner = (-0.5, 0.2)
    for k in range(1, 26):
        r = 0.75 * k ** 1.08
        n = int(np.clip(2 * math.pi * r / 0.16, 24, 170))
        t = np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(0, 0.1)
        x = scanner[0] + r * np.cos(t) + rng.normal(0, 0.035, n)
        y = scanner[1] + r * np.sin(t) + rng.normal(0, 0.035, n)
        m = (x > -10.2) & (x < 9.0) & (y > -4.6) & (y < 4.4)
        m &= (x - tank_c[0]) ** 2 + (y - tank_c[1]) ** 2 > tank_r**2
        m &= ~((abs(x - pump_c[0] - 0.6) < 1.8) & (abs(y - pump_c[1]) < 0.9))
        parts["ground"].append(np.c_[x[m], y[m], np.zeros(m.sum())])

    # storage tank with stiffening bands
    parts["tank"] += [cyl_v(*tank_c, tank_r, 0, 5.2, 6.5), disk(*tank_c, 5.2, tank_r, 4.5)]
    for zb in (1.3, 2.6, 3.9):
        parts["tank"].append(cyl_v(*tank_c, tank_r + 0.03, zb - 0.04, zb + 0.04, 70))

    # pipe rack + connections to the tank and the pump
    z = 2.6
    parts["pipe"] += [
        pipe((-9.5, -2.2, z), (3.1, -2.2, z), 0.28, 11),
        pipe((-9.5, -3.0, z), (3.1, -3.0, z), 0.22, 11),
        pipe((3.1, -2.2, z), (3.1, 0.2, z), 0.24, 11),
        pipe((3.1, 0.2, z), (3.5, 0.2, z), 0.24, 11),
        pipe((pump_c[0], pump_c[1], 1.0), (pump_c[0], pump_c[1], z), 0.2, 12),
        pipe((pump_c[0], pump_c[1], z), (pump_c[0], -2.2, z), 0.2, 12),
    ]
    for x in (-8.5, -5.5, -2.5, 0.5):
        parts["structure"] += [
            pipe((x, -2.6, 0), (x, -2.6, z - 0.3), 0.09, 30),
            pipe((x, -3.5, z - 0.3), (x, -1.7, z - 0.3), 0.07, 30),
        ]
    parts["structure"].append(box((pump_c[0] + 0.6, pump_c[1], 0.06), (3.4, 1.6, 0.12), 7))

    # pump body + motor
    parts["pump"] += [
        box((pump_c[0], pump_c[1], 0.62), (2.0, 1.2, 1.0), 13),
        pipe((pump_c[0] + 1.0, pump_c[1], 0.62), (pump_c[0] + 2.3, pump_c[1], 0.62), 0.42, 13),
    ]
    return {k: jitter(np.vstack(v)) for k, v in parts.items()}


# ---------- scene B: voxelized breast DCE-MRI volume ----------
SHADES = ["#5d6670", "#77818b", "#919aa4", "#acb4bd", "#c9cfd6"]  # dark -> bright tissue


def smooth_noise(P, waves, freq):
    """Sum of random plane waves: smooth 3D noise with roughly unit variance."""
    acc = np.zeros(P.shape[:-1])
    for _ in range(waves):
        acc += np.sin(P @ rng.normal(0, freq, 3) + rng.uniform(0, 2 * math.pi))
    return acc / math.sqrt(waves / 2)


def breast_volume():
    """Voxel grid of one breast on a block of chest wall, with the half facing the viewer
    cut away (y > 0 removed) so the inside shows, like a sagittal cross-section."""
    NX, NY, NZ = 31, 13, 25
    XW = 5  # chest wall thickness
    RX, RY, RZ = 22.0, 12.5, 11.0  # breast half-ellipsoid
    ZC0, DROOP = 13.0, 0.12  # center height at the chest wall, and how much it sags outward
    i, j, k = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing="ij")
    x, y, z = i + 0.5, j + 0.5 - NY, k + 0.5
    zc = ZC0 - DROOP * (x - XW)
    rho = np.sqrt(((x - XW) / RX) ** 2 + (y / RY) ** 2 + ((z - zc) / RZ) ** 2)
    wall = x < XW
    dome = (x >= XW) & (rho <= 1)
    tip = XW + RX
    nipple = (x >= tip - 1) & (x < tip + 1.8) & (y**2 + (z - (ZC0 - DROOP * RX)) ** 2 <= 2.0**2)
    occ = wall | dome | nipple

    P = np.stack([x, y, z], -1)
    lob, fine, edge = smooth_noise(P, 10, 0.35), smooth_noise(P, 10, 0.7), smooth_noise(P, 6, 0.6)
    # distance from the duct axis (chest wall -> nipple), relative to the local breast radius
    t = np.clip((x - XW) / RX, 0, 1)
    local_r = np.sqrt(np.clip(1 - t**2, 0.05, 1))
    d_axis = np.sqrt((y / RY) ** 2 + ((z - zc) / RZ) ** 2) / local_r

    shade = np.zeros(occ.shape, int)
    shade[wall] = np.where(fine[wall] > 0.8, 1, 0)  # thorax
    shade[wall & (x >= XW - 2)] = 1  # pectoral muscle
    shade[dome] = 1  # fat
    gland = dome & (0.75 * lob + 0.35 * fine + 2.0 * (1 - d_axis) - 0.9 > 0)
    gland &= (x - XW > 1.5) & (rho < 0.86)  # retromammary and subcutaneous fat stay dark
    shade[gland] = 3
    shade[dome & (rho > 1 - 1.2 / RZ)] = 2  # skin
    shade[nipple] = 2
    lc, lr = np.array([XW + 9.5, -0.8, ZC0 - DROOP * 9.5 + 0.6]), np.array([3.3, 2.8, 3.0])
    lesion = occ & (np.sqrt((((P - lc) / lr) ** 2).sum(-1)) <= 1 + 0.2 * edge)  # irregular border
    shade[lesion] = 4  # enhancing lesion is bright in DCE

    pad = np.pad(occ, ((0, 1), (0, 1), (0, 1)))
    open_x, open_y, open_z = ~pad[1:, :-1, :-1], ~pad[:-1, 1:, :-1], ~pad[:-1, :-1, 1:]
    vis = occ & (open_x | open_y | open_z)  # only voxels with a face toward the camera
    idx = np.argwhere(vis)
    corner = idx.astype(float) - [0, NY, 0]  # min corner of each voxel
    return corner, shade[vis], lesion[vis] & open_y[vis]


# ---------- projection + SVG ----------
def camera(az, el):
    """Orthographic camera: az rotates the ground plane, el is the elevation above the horizon.
    Returns view(P) -> (screen uv, ground-plane depth w), and depth(P), the true distance
    toward the viewer used to sort voxels."""
    ca, sa = math.cos(math.radians(az)), math.sin(math.radians(az))
    ce, se = math.cos(math.radians(el)), math.sin(math.radians(el))

    def view(P):
        P = np.atleast_2d(P)
        u = P[:, 0] * ca - P[:, 1] * sa
        w = P[:, 0] * sa + P[:, 1] * ca
        return np.c_[u, w * se - P[:, 2] * ce], w

    def depth(P):
        P = np.atleast_2d(P)
        return P[:, 0] * sa * ce + P[:, 1] * ca * ce + P[:, 2] * se

    view.depth = depth
    return view


def fit(points_2d, box_):
    lo, hi = points_2d.min(0), points_2d.max(0)
    x0, y0, x1, y1 = box_
    s = min((x1 - x0) / (hi - lo)[0], (y1 - y0) / (hi - lo)[1])
    off = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) - s * (lo + hi) / 2
    return s, off


BINS = [(2.0, 0.5), (2.5, 0.75), (3.0, 0.95)]  # LiDAR (dot size, opacity) far -> near


def dots(uv):
    return "".join(f"M{u:.1f} {v:.1f}h0" for u, v in uv)


def lidar_layer(parts, proj, cuts, colored):
    out = []
    for b, (sw, op) in enumerate(BINS):
        for name, P in parts.items():
            if colored and name == "ground":
                continue  # ground stays gray: the gray layer already shows it
            uv, depth = proj(P)
            m = np.digitize(depth, cuts) == b
            if m.any():
                col = COLORS[name] if colored else GRAY
                o = op * (0.5 if name == "ground" else 1)
                out.append(f'<path stroke="{col}" stroke-width="{sw}" stroke-opacity="{o:.2f}" d="{dots(uv[m])}"/>')
    return "\n".join(out)


def mix(hex_, target, a):
    c = np.array([int(hex_[i : i + 2], 16) for i in (1, 3, 5)])
    t = np.array([255, 255, 255]) if target == "white" else np.zeros(3)
    return "#" + "".join(f"{int(round(v)):02x}" for v in c * (1 - a) + t * a)


def main(path):
    # LiDAR
    A = plant()
    view_a = camera(28, 24)
    s, off = fit(np.vstack([view_a(p)[0] for p in A.values()]), (14, 10, 610, 266))
    pa = lambda P: (view_a(P)[0] * s + off, view_a(P)[1])
    cuts = np.quantile(np.concatenate([pa(p)[1] for p in A.values()]), [1 / 3, 2 / 3])

    # MRI voxels
    corner, shade, les_face = breast_volume()
    view_b = camera(20, 19)
    cube = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], float)
    all_corners = (corner[:, None, :] + cube[None]).reshape(-1, 3)
    sb, offb = fit(view_b(all_corners)[0], (664, 12, 986, 262))
    uv0, depth = view_b(corner)[0], view_b.depth(corner + 0.5)
    uv0 = uv0 * sb + offb
    order = np.argsort(depth, kind="stable")  # painter's algorithm: far voxels first

    def face(pts):
        q = view_b(np.array(pts, float))[0] * sb
        q -= view_b(np.zeros(3))[0] * sb
        return "M" + "L".join(f"{a:.2f} {b:.2f}" for a, b in q) + "Z"

    top = face([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    fx = face([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)])
    fy = face([(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)])
    voxel_defs = "".join(
        f'<g id="v{n}"><path fill="{mix(c, "white", 0.18)}" d="{top}"/>'
        f'<path fill="{mix(c, "black", 0.22)}" d="{fx}"/><path fill="{c}" d="{fy}"/></g>'
        for n, c in enumerate(SHADES)
    )
    voxels = "".join(f'<use href="#v{shade[n]}" x="{uv0[n, 0]:.1f}" y="{uv0[n, 1]:.1f}"/>' for n in order)
    lesion = "".join(f'<use href="#ly" x="{uv0[n, 0]:.1f}" y="{uv0[n, 1]:.1f}"/>' for n in order if les_face[n])

    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
    TA, TB = "9s", "7s"  # different periods so the two "models" drift in and out of sync
    ys, ye = 8, 268  # vertical extent of the MRI sweep

    def legend(x, y, items):
        out, cx = [], x
        for name, label in items:
            out.append(f'<circle cx="{cx + 4}" cy="{y - 4}" r="4" fill="{COLORS[name]}"/>')
            out.append(f'<text x="{cx + 13}" y="{y}">{label}</text>')
            cx += 22 + 7.3 * len(label)
        return "".join(out)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">
<title id="t">3D segmentation: a LiDAR point cloud of an industrial plant and a voxelized breast MRI volume, each revealed by its own scan line</title>
<defs>
<clipPath id="ra"><rect x="0" y="0" width="{SPLIT}" height="{H}">
<animate attributeName="width" values="0;{SPLIT};{SPLIT}" keyTimes="0;0.45;1" dur="{TA}" repeatCount="indefinite"/></rect></clipPath>
<clipPath id="rb"><rect x="{SPLIT}" y="{ys}" width="{W - SPLIT}" height="{ye - ys}">
<animate attributeName="height" values="0;{ye - ys};{ye - ys}" keyTimes="0;0.5;1" dur="{TB}" repeatCount="indefinite"/></rect></clipPath>
{voxel_defs}
<path id="ly" fill="{COLORS['lesion']}" d="{fy}"/>
</defs>

<g fill="none" stroke-linecap="round">
{lidar_layer(A, pa, cuts, colored=False)}
<g clip-path="url(#ra)">
<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.82;0.95;1" dur="{TA}" repeatCount="indefinite"/>
{lidar_layer(A, pa, cuts, colored=True)}
</g>
</g>
<line x1="0" y1="8" x2="0" y2="270" stroke="{COLORS['tank']}" stroke-width="1.5" opacity="0">
<animate attributeName="x1" values="0;{SPLIT};{SPLIT}" keyTimes="0;0.45;1" dur="{TA}" repeatCount="indefinite"/>
<animate attributeName="x2" values="0;{SPLIT};{SPLIT}" keyTimes="0;0.45;1" dur="{TA}" repeatCount="indefinite"/>
<animate attributeName="opacity" values="0;0.8;0.8;0;0" keyTimes="0;0.03;0.42;0.46;1" dur="{TA}" repeatCount="indefinite"/>
</line>

<g stroke="#000" stroke-opacity="0.13" stroke-width="0.4" stroke-linejoin="round">{voxels}</g>
<g clip-path="url(#rb)" stroke="#000" stroke-opacity="0.13" stroke-width="0.4" fill-opacity="0.92">
<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.8;0.93;1" dur="{TB}" repeatCount="indefinite"/>
{lesion}
</g>
<line x1="{SPLIT + 12}" y1="{ys}" x2="{W - 8}" y2="{ys}" stroke="{COLORS['lesion']}" stroke-width="1.5" opacity="0">
<animate attributeName="y1" values="{ys};{ye};{ye}" keyTimes="0;0.5;1" dur="{TB}" repeatCount="indefinite"/>
<animate attributeName="y2" values="{ys};{ye};{ye}" keyTimes="0;0.5;1" dur="{TB}" repeatCount="indefinite"/>
<animate attributeName="opacity" values="0;0.8;0.8;0;0" keyTimes="0;0.03;0.47;0.51;1" dur="{TB}" repeatCount="indefinite"/>
</line>

<line x1="{SPLIT}" y1="18" x2="{SPLIT}" y2="300" stroke="{GRAY}" stroke-opacity="0.3" stroke-dasharray="2 5"/>
<g font-family="{mono}" font-size="12" fill="{GRAY}">
<text x="20" y="290">lidar point cloud — equipment segmentation</text>
<text x="652" y="290">breast dce-mri — lesion segmentation</text>
{legend(20, 312, [("tank", "tank"), ("pipe", "piping"), ("pump", "pump"), ("structure", "structure")])}
{legend(652, 312, [("lesion", "lesion")])}
</g>
</svg>
"""
    with open(path, "w") as f:
        f.write(svg)
    print(f"wrote {path}: {len(svg) / 1024:.0f} KB, {sum(len(p) for p in A.values())} lidar points, "
          f"{len(corner)} visible voxels ({int(les_face.sum())} lesion faces on the cut)")


if __name__ == "__main__":
    main(sys.argv[1])
