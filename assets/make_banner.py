"""Generate the profile banner: a LiDAR point cloud of a plant and a DCE-MRI volume,
drawn in gray, with a scan line that sweeps across and reveals the segmentation colors.

Pure static SVG (SMIL animation), so it needs no GitHub Actions or external service.
Usage: python make_banner.py OUT.svg
"""
import math
import sys

import numpy as np

rng = np.random.default_rng(7)
W, H = 1000, 320
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


# ---------- scene B: stack of DCE-MRI slices with a lesion ----------
def mri():
    zs = [0, 1.5, 3.0, 4.5, 6.0]
    lc, lr = np.array([0.9, -0.4, 3.0]), np.array([1.4, 1.1, 2.3])
    tissue, lesion = [], []
    for z in zs:
        f = 1 - 0.1 * ((z - 3) / 3) ** 2
        ax, ay = 3.6 * f, 2.6 * f
        g = np.arange(-4, 4.01, 0.34)
        X, Y = (a.ravel() + rng.normal(0, 0.05, a.size) for a in np.meshgrid(g, g))
        inside = (X / ax) ** 2 + (Y / ay) ** 2 <= 1
        tex = 0.5 + 0.5 * np.sin(1.4 * X + 0.6 * z) * np.cos(1.2 * Y - 0.5 * z)
        keep = inside & (rng.uniform(0, 1, X.size) < 0.2 + 0.5 * tex)
        in_les = ((X - lc[0]) / lr[0]) ** 2 + ((Y - lc[1]) / lr[1]) ** 2 + ((z - lc[2]) / lr[2]) ** 2 <= 1
        tissue.append(np.c_[X[keep & ~in_les], Y[keep & ~in_les], np.full((keep & ~in_les).sum(), z)])
        t = rng.uniform(0, 2 * math.pi, 90)  # body outline
        tissue.append(np.c_[ax * np.cos(t), ay * np.sin(t), np.full(90, z)])
        # lesion cross-section on this slice, sampled denser
        k = 1 - ((z - lc[2]) / lr[2]) ** 2
        if k > 0:
            a, b = lr[0] * math.sqrt(k), lr[1] * math.sqrt(k)
            n = int(math.pi * a * b * 34)
            rr, tt = np.sqrt(rng.uniform(0, 1, n)), rng.uniform(0, 2 * math.pi, n)
            lesion.append(np.c_[lc[0] + a * rr * np.cos(tt), lc[1] + b * rr * np.sin(tt), np.full(n, z)])
    frame = []  # slice outlines + vertical edges of the volume
    sq = [(-4, -4), (4, -4), (4, 4), (-4, 4)]
    for z in zs:
        frame += [((*sq[i], z), (*sq[(i + 1) % 4], z)) for i in range(4)]
    frame += [((*c, zs[0]), (*c, zs[-1])) for c in sq]
    pts = {"tissue": jitter(np.vstack(tissue), 0.02), "lesion": jitter(np.vstack(lesion), 0.02)}
    return pts, frame


# ---------- projection + SVG ----------
def camera(az, el):
    """Orthographic camera: az rotates the ground plane, el is the elevation above the horizon."""
    ca, sa, ce, se = math.cos(math.radians(az)), math.sin(math.radians(az)), math.cos(math.radians(el)), math.sin(math.radians(el))

    def view(P):
        P = np.atleast_2d(P)
        u = P[:, 0] * ca - P[:, 1] * sa
        w = P[:, 0] * sa + P[:, 1] * ca  # toward the viewer
        return np.c_[u, w * se - P[:, 2] * ce], w

    return view


def fit(parts, extra, box_, iso):
    allp = np.vstack([iso(p)[0] for p in parts.values()] + ([iso(np.array(extra))[0]] if extra else []))
    lo, hi = allp.min(0), allp.max(0)
    x0, y0, x1, y1 = box_
    s = min((x1 - x0) / (hi - lo)[0], (y1 - y0) / (hi - lo)[1])
    off = np.array([(x0 + x1) / 2, (y0 + y1) / 2]) - s * (lo + hi) / 2
    return lambda P: (iso(P)[0] * s + off, iso(P)[1])


BINS = [(2.0, 0.5), (2.5, 0.75), (3.0, 0.95)]  # (dot size, opacity) far -> near


def dots(uv):
    return "".join(f"M{u:.1f} {v:.1f}h0" for u, v in uv)


def layer(scenes, colored):
    out = []
    for b, (sw, op) in enumerate(BINS):
        for parts, proj, cuts, bg in scenes:
            for name, P in parts.items():
                if colored and name in bg:
                    continue  # background stays gray: the gray layer already shows it
                uv, depth = proj(P)
                m = np.digitize(depth, cuts) == b
                if not m.any():
                    continue
                col = COLORS[name] if colored else GRAY
                o = op * (0.5 if (name in bg) else 1)
                out.append(f'<path stroke="{col}" stroke-width="{sw}" stroke-opacity="{o:.2f}" d="{dots(uv[m])}"/>')
    return "\n".join(out)


def main(path):
    A = plant()
    B, frame = mri()
    pa = fit(A, None, (14, 10, 610, 266), camera(28, 24))
    pb = fit(B, [p for e in frame for p in e], (660, 10, 986, 266), camera(38, 26))
    cut = lambda parts, proj: np.quantile(np.concatenate([proj(p)[1] for p in parts.values()]), [1 / 3, 2 / 3])
    scenes = [(A, pa, cut(A, pa), {"ground"}), (B, pb, cut(B, pb), {"tissue"})]

    frame_svg = "".join(
        f"M{a[0]:.1f} {a[1]:.1f}L{b[0]:.1f} {b[1]:.1f}"
        for a, b in (pb(np.array(e))[0] for e in frame)
    )
    T, sweep = "9s", 'keyTimes="0;0.45;1"'
    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

    def legend(x, y, items):
        s, cx = [], x
        for name, label in items:
            s.append(f'<circle cx="{cx + 4}" cy="{y - 4}" r="4" fill="{COLORS[name]}"/>')
            s.append(f'<text x="{cx + 13}" y="{y}">{label}</text>')
            cx += 22 + 7.3 * len(label)
        return "".join(s)

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">
<title id="t">3D segmentation: a LiDAR point cloud of an industrial plant and a contrast MRI volume, revealed by a sweeping scan line</title>
<defs><clipPath id="reveal"><rect x="0" y="0" width="{W}" height="{H}">
<animate attributeName="width" values="0;{W};{W}" {sweep} dur="{T}" repeatCount="indefinite"/></rect></clipPath></defs>
<g fill="none" stroke-linecap="round">
<path d="{frame_svg}" stroke="{GRAY}" stroke-opacity="0.22" stroke-width="1"/>
{layer(scenes, colored=False)}
<g clip-path="url(#reveal)">
<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;0.82;0.95;1" dur="{T}" repeatCount="indefinite"/>
{layer(scenes, colored=True)}
</g>
</g>
<line x1="0" y1="8" x2="0" y2="270" stroke="{COLORS['tank']}" stroke-width="1.5" opacity="0">
<animate attributeName="x1" values="0;{W};{W}" {sweep} dur="{T}" repeatCount="indefinite"/>
<animate attributeName="x2" values="0;{W};{W}" {sweep} dur="{T}" repeatCount="indefinite"/>
<animate attributeName="opacity" values="0;0.8;0.8;0;0" keyTimes="0;0.03;0.42;0.46;1" dur="{T}" repeatCount="indefinite"/>
</line>
<g font-family="{mono}" font-size="12" fill="{GRAY}">
<text x="20" y="290">lidar point cloud — equipment segmentation</text>
<text x="650" y="290">dce-mri volume — lesion segmentation</text>
{legend(20, 312, [("tank", "tank"), ("pipe", "piping"), ("pump", "pump"), ("structure", "structure")])}
{legend(650, 312, [("lesion", "lesion")])}
</g>
</svg>
"""
    with open(path, "w") as f:
        f.write(svg)
    n = sum(len(p) for p in A.values()) + sum(len(p) for p in B.values())
    print(f"wrote {path}: {len(svg) / 1024:.0f} KB, {n} points")


if __name__ == "__main__":
    main(sys.argv[1])
