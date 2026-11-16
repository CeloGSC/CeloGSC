"""Generate the profile banner.

Left: a LiDAR point cloud of a small plant. A vertical scan line sweeps left to right and
colors the segmented equipment.
Right: a breast DCE-MRI viewer stepping through axial slices. Where the tumor shows up it is
segmented (red outline + fill), and next to the viewer it is rebuilt in 3D, one layer per
slice, as the scan passes through it.
Both panels share one clock: they scan together, hold, fade out together and restart.

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


# ---------- scene B: breast DCE-MRI, slice by slice ----------
NU, NV, NS = 76, 42, 20  # slice width, height (pixels) and number of axial slices
NLEV = 9  # gray levels drawn on the black viewer (level 0 = black, not drawn)
LEVELS = [None] + ["#%02x%02x%02x" % ((g,) * 3) for g in (int(18 + 228 * (k / NLEV) ** 0.95) for k in range(1, NLEV + 1))]


def smooth_noise(P, waves, freq):
    """Sum of random plane waves: smooth noise with roughly unit variance."""
    acc = np.zeros(P.shape[:-1])
    for _ in range(waves):
        acc += np.sin(P @ rng.normal(0, freq, P.shape[-1]) + rng.uniform(0, 2 * math.pi))
    return acc / math.sqrt(waves / 2)


def breast_mri():
    """Axial slices of a fat-suppressed, contrast-enhanced breast MRI, displayed anterior-up:
    two breasts on the chest wall (skin, nipple, suppressed fat, fibroglandular tissue,
    enhancing vessels), pectoral muscle, sternum, heart with contrast, dark lungs, and one
    spiculated, rim-enhancing tumor. Returns gray levels[s, v, u] and the tumor mask[s, v, u]."""
    s, v, u = np.meshgrid(np.arange(NS), np.arange(NV) + 0.5, np.arange(NU) + 0.5, indexing="ij")
    P = np.stack([u, v, s * 2.0], -1)
    lob, fine, edge = smooth_noise(P, 12, 0.2), smooth_noise(P, 12, 0.45), smooth_noise(P, 8, 0.35)
    q = (s - (NS - 1) / 2) / ((NS - 1) / 2)  # -1 (top slice) .. 1 (bottom slice)
    prof = np.sqrt(np.clip(1 - (0.85 * q) ** 2, 0, 1))

    e_th = ((u - 38) / 39) ** 2 + ((v - 50) / 23) ** 2
    thorax = e_th <= 1
    I = np.zeros(u.shape)
    breasts = np.zeros(u.shape, bool)
    bases = []
    for ub in (19.5, 56.5):
        vb = 50 - 23 * math.sqrt(1 - ((ub - 38) / 39) ** 2) + 0.5  # chest surface under this breast
        bases.append((ub, vb))
        rw, rh = 14.5 * (0.72 + 0.28 * prof), 4 + 24 * prof**0.8
        tip = np.clip((vb - v) / rh, 0, 1)  # 0 at the chest wall, 1 at the nipple
        width = rw * (1 - 0.2 * tip**2)
        d = ((np.abs(u - ub) / width) ** 2.2 + (np.clip(vb - v, 0, None) / rh) ** 2.2) ** (1 / 2.2)
        chest = 50 - 23 * np.sqrt(np.clip(1 - ((u - 38) / 39) ** 2, 0, 1))
        inside = (d <= 1) & (v <= chest + 0.6)  # the base reaches down to the curved chest wall
        nipple = (prof > 0.95) & (np.hypot((u - ub) / 1.3, v - (vb - rh - 0.3)) <= 1.0)
        breasts |= inside | nipple
        I = np.where(inside, 0.15 + 0.015 * lob, I)  # suppressed fat: smooth, dark
        d_axis = np.abs(u - ub) / width
        gland = inside & (0.9 * lob + 0.25 * fine + 2.2 * (1 - d_axis * 1.25) - 0.35 * tip - 1.0 > 0)
        gland &= (d < 0.86) & (v < chest - 2.5) & (tip < 0.9)  # keep subcutaneous and retromammary fat
        I = np.where(gland, 0.46 + 0.05 * lob + 0.03 * fine, I)  # fibroglandular tissue, lacy margins
        I = np.where(nipple, 0.34, I)
    body = thorax | breasts

    # thorax: pectoral muscle, sternum, heart with contrast, dark lungs
    I = np.where(thorax, 0.0 + 0.16 * (fine > 1.75), I)  # lungs: black, a few vessels
    I = np.where(thorax & (e_th > 0.83), 0.26 + 0.015 * fine, I)  # pectoral muscle
    I = np.where(thorax & (((u - 38) / 2.5) ** 2 + ((v - 28.8) / 1.2) ** 2 <= 1), 0.3, I)  # sternum
    hs = np.clip(1 - 0.8 * np.abs(q - 0.4), 0.15, 1)
    e_h = ((u - 34) / (13 * hs)) ** 2 + ((v - 41.5) / (8 * hs)) ** 2
    I = np.where(thorax & (e_th <= 0.83) & (e_h <= 1), 0.36 + 0.02 * fine, I)  # myocardium
    for cu, cv, r in ((30, 41.5, 0.45), (38.5, 42, 0.35)):  # ventricles: blood pool with contrast
        I = np.where(thorax & ((((u - cu) / (13 * hs)) ** 2 + ((v - cv) / (8 * hs)) ** 2) <= r), 0.82, I)

    # enhancing vessels: short bright segments where 3D curves cross each slice
    for ub, vb in bases:
        for _ in range(8):
            t = np.linspace(0, 1, 400)
            u0, ph, f = ub + rng.uniform(-9, 9), rng.uniform(0, 6.3), rng.uniform(0.6, 1.6)
            uc = u0 + (ub - u0) * 0.5 * t + 1.6 * np.sin(2 * math.pi * f * t + ph)
            vc = vb - 1.5 - t * rng.uniform(9, 19)
            sc = rng.uniform(1, NS - 2) + rng.choice([-1, 1]) * rng.uniform(7, 12) * t
            for k in range(NS):
                m = np.abs(sc - k) < 0.3
                iu, iv = uc[m].astype(int), vc[m].astype(int)
                ok = (iu >= 0) & (iu < NU) & (iv >= 0) & (iv < NV)
                iu, iv = iu[ok], iv[ok]
                keep = breasts[k, iv, iu]
                I[k, iv[keep], iu[keep]] = 0.78

    # skin
    pad = np.pad(body, ((0, 0), (1, 1), (1, 1)))
    skin = body & ~(pad[:, :-2, 1:-1] & pad[:, 2:, 1:-1] & pad[:, 1:-1, :-2] & pad[:, 1:-1, 2:])
    I = np.where(skin & breasts, 0.3, I)

    # tumor: spiculated mass with rim enhancement
    tc, tr = np.array([17.0, 14.5, 10.0 * 2.0]), np.array([4.6, 4.0, 3.0 * 2.0])
    rn = np.sqrt((((P - tc) / tr) ** 2).sum(-1))
    tumor = breasts & (rn <= 1 + 0.2 * edge)
    theta = np.arctan2(v - tc[1], u - tc[0])
    spike = np.maximum(0, np.cos(7 * theta + 0.5 * s)) ** 14
    spicules = breasts & ~tumor & (rn <= 1 + 0.75 * spike) & (spike > 0.3)
    I = np.where(spicules, 0.55, I)
    I = np.where(tumor, np.where(rn > 0.62, 0.95, 0.66 + 0.1 * fine), I)

    I *= 0.88 + 0.18 * (1 - v / NV)  # receive-coil sensitivity: brighter near the breasts
    I = np.where(body, I + rng.normal(0, 0.012, I.shape), 0)  # image grain
    lev = np.clip(np.rint(I * NLEV), 0, NLEV).astype(int)
    return lev, tumor


def runs(mask):
    """Row runs of a 2D mask as a compact path of 1-pixel-tall strokes."""
    out = []
    for v, row in enumerate(mask):
        d = np.diff(np.r_[0, row.astype(np.int8), 0])
        a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        for k in range(len(a)):
            out.append(f"M{a[k]} {v}.5h{b[k] - a[k]}" if k == 0 else f"m{a[k] - b[k - 1]} 0h{b[k] - a[k]}")
    return "".join(out)


def discrete(attr, values, times, dur):
    kt = ";".join(f"{t:.4f}".rstrip("0").rstrip(".") for t in times)
    return f'<animate attributeName="{attr}" values="{";".join(values)}" keyTimes="{kt}" calcMode="discrete" dur="{dur}" repeatCount="indefinite"/>'


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
    # one shared clock: both scans run 0 -> SCAN, hold, then fade out together
    T, SCAN, HOLD, FADE = 10, 0.6, 0.86, 0.96
    TD = f"{T}s"

    # LiDAR (points unchanged)
    A = plant()
    view_a = camera(28, 24)
    s, off = fit(np.vstack([view_a(p)[0] for p in A.values()]), (14, 10, 610, 266))
    pa = lambda P: (view_a(P)[0] * s + off, view_a(P)[1])
    cuts = np.quantile(np.concatenate([pa(p)[1] for p in A.values()]), [1 / 3, 2 / 3])

    # MRI slices
    lev, tumor = breast_mri()
    t_on = [i * SCAN / NS for i in range(NS)]
    t_off = t_on[1:] + [1.0]
    hero = int(np.argmax(tumor.sum((1, 2))))  # shown when animation is unsupported
    tumor_slices = [i for i in range(NS) if tumor[i].any()]
    red = COLORS["lesion"]

    PX, VX, VY = 2.8, 650, 86  # pixel size and top-left of the image
    IW, IH = NU * PX, NV * PX
    slices = []
    for i in range(NS):
        if i == 0:
            anim = discrete("visibility", ["visible", "hidden"], [0, t_off[0]], TD)
        elif i == NS - 1:
            anim = discrete("visibility", ["hidden", "visible"], [0, t_on[i]], TD)
        else:
            anim = discrete("visibility", ["hidden", "visible", "hidden"], [0, t_on[i], t_off[i]], TD)
        img = "".join(f'<path stroke="{LEVELS[l]}" d="{runs(lev[i] == l)}"/>' for l in range(1, NLEV + 1) if (lev[i] == l).any())
        tag = ""
        if tumor[i].any():
            m = tumor[i]
            pm = np.pad(m, 1)
            border = m & ~(pm[:-2, 1:-1] & pm[2:, 1:-1] & pm[1:-1, :-2] & pm[1:-1, 2:])
            img += f'<path stroke="{red}" stroke-opacity="0.32" d="{runs(m & ~border)}"/><path stroke="{red}" d="{runs(border)}"/>'
            tag = (f'<circle cx="{VX + IW - 34}" cy="{VY - 12}" r="3" fill="{red}"/>'
                   f'<text x="{VX + IW}" y="{VY - 9}" fill="{red}" text-anchor="end">tumor</text>')
        slices.append(
            f'<g visibility="{"visible" if i == hero else "hidden"}">{anim}'
            f'<g transform="translate({VX} {VY}) scale({PX})" fill="none" stroke-width="1.04">{img}</g>'
            f'<text x="{VX}" y="{VY - 9}">slice {i + 1:02d}/{NS}</text>{tag}</g>'
        )
    overlay = (f'<text x="{VX}" y="{VY + IH + 13}">T1 FS +C</text>'
               f'<text x="{VX + IW}" y="{VY + IH + 13}" text-anchor="end">AX</text>'
               f'<text x="{VX - 5}" y="{VY + IH / 2 + 3}" text-anchor="end">R</text>'
               f'<text x="{VX + IW + 5}" y="{VY + IH / 2 + 3}">L</text>')

    # 3D reconstruction of the tumor, one layer per detected slice
    vv, uu = np.nonzero(tumor.any(0))
    u0, v0 = uu.min() - 2, vv.min() - 2
    BU, BV = uu.max() + 3 - u0, vv.max() + 3 - v0
    ZS = 1.3  # slices are thicker than the in-plane pixels, as in a real DCE acquisition
    view_b = camera(30, 24)
    box_c = np.array([[a, b, c] for a in (0, BU) for b in (0, BV) for c in (0, NS * ZS)], float)
    sb, offb = fit(view_b(box_c)[0], (884, 26, 990, 256))
    P2 = lambda P: view_b(np.asarray(P, float))[0] * sb + offb

    def face(pts):
        q = view_b(np.array(pts, float))[0] * sb - view_b(np.zeros(3))[0] * sb
        return "M" + "L".join(f"{a:.2f} {b:.2f}" for a, b in q) + "Z"

    cube = (f'<g id="tc"><path fill="{mix(red, "white", 0.25)}" d="{face([(0, 0, ZS), (1, 0, ZS), (1, 1, ZS), (0, 1, ZS)])}"/>'
            f'<path fill="{mix(red, "black", 0.25)}" d="{face([(1, 0, 0), (1, 1, 0), (1, 1, ZS), (1, 0, ZS)])}"/>'
            f'<path fill="{red}" d="{face([(0, 1, 0), (1, 1, 0), (1, 1, ZS), (0, 1, ZS)])}"/></g>')
    layers = []
    for i in sorted(tumor_slices, reverse=True):  # bottom slice first: painter's order from above
        z = (NS - 1 - i) * ZS
        vs, us = np.nonzero(tumor[i])
        pts = np.c_[us - u0, vs - v0, np.full(len(us), z)].astype(float)
        order = np.argsort(view_b.depth(pts + 0.5))
        uv = P2(pts)
        uses = "".join(f'<use href="#tc" x="{uv[n, 0]:.1f}" y="{uv[n, 1]:.1f}"/>' for n in order)
        layers.append(f'<g>{discrete("visibility", ["hidden", "visible"], [0, t_on[i]], TD)}{uses}</g>')

    edges = [(a, b) for a in box_c for b in box_c if (np.abs(a - b) > 0).sum() == 1 and tuple(a) < tuple(b)]
    wire = "".join(f"M{P2(a)[0, 0]:.1f} {P2(a)[0, 1]:.1f}L{P2(b)[0, 0]:.1f} {P2(b)[0, 1]:.1f}" for a, b in edges)
    plane_d = "M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in P2([(0, 0, 0), (BU, 0, 0), (BU, BV, 0), (0, BV, 0)])) + "Z"
    dy = lambda i: (P2([(0, 0, (NS - 1 - i) * ZS + ZS / 2)]) - P2([(0, 0, 0)]))[0, 1]
    plane_anim = (f'<animateTransform attributeName="transform" type="translate" calcMode="discrete" '
                  f'values="{";".join(f"0 {dy(i):.1f}" for i in range(NS))}" '
                  f'keyTimes="{";".join(f"{t:.4f}" for t in t_on)}" dur="{TD}" repeatCount="indefinite"/>')
    fade = f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;{HOLD};{FADE};1" dur="{TD}" repeatCount="indefinite"/>'

    def legend(x, y, items):
        out, cx = [], x
        for color, label in items:
            out.append(f'<circle cx="{cx + 4}" cy="{y - 4}" r="4" fill="{color}"/>')
            out.append(f'<text x="{cx + 13}" y="{y}">{label}</text>')
            cx += 22 + 7.3 * len(label)
        return "".join(out)

    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"
    sweep = f'keyTimes="0;{SCAN};1" dur="{TD}" repeatCount="indefinite"'
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-labelledby="t">
<title id="t">3D segmentation: a LiDAR point cloud of an industrial plant being segmented, and a breast MRI scanned slice by slice, where the tumor is segmented and rebuilt in 3D</title>
<defs>
<clipPath id="ra"><rect x="0" y="0" width="{SPLIT}" height="{H}">
<animate attributeName="width" values="0;{SPLIT};{SPLIT}" {sweep}/></rect></clipPath>
{cube}
</defs>

<g fill="none" stroke-linecap="round">
{lidar_layer(A, pa, cuts, colored=False)}
<g clip-path="url(#ra)">{fade}
{lidar_layer(A, pa, cuts, colored=True)}
</g>
</g>
<line x1="0" y1="8" x2="0" y2="270" stroke="{COLORS['tank']}" stroke-width="1.5" opacity="0">
<animate attributeName="x1" values="0;{SPLIT};{SPLIT}" {sweep}/>
<animate attributeName="x2" values="0;{SPLIT};{SPLIT}" {sweep}/>
<animate attributeName="opacity" values="0;0.8;0.8;0;0" keyTimes="0;0.02;{SCAN - 0.03:.2f};{SCAN + 0.01:.2f};1" dur="{TD}" repeatCount="indefinite"/>
</line>

<rect x="{VX - 16}" y="{VY - 26}" width="{IW + 32}" height="{IH + 46}" rx="6" fill="#0b0e12" stroke="#30363d"/>
<g font-family="{mono}" font-size="9" fill="{GRAY}">
{chr(10).join(slices)}
{overlay}
</g>

<path d="{wire}" fill="none" stroke="{GRAY}" stroke-opacity="0.4" stroke-width="0.8"/>
<g stroke="#000" stroke-opacity="0.15" stroke-width="0.4" stroke-linejoin="round">{fade}
{chr(10).join(layers)}
</g>
<path d="{plane_d}" fill="{GRAY}" fill-opacity="0.12" stroke="{GRAY}" stroke-opacity="0.8" stroke-width="0.8" transform="translate(0 {dy(hero):.1f})">
{plane_anim}
{discrete("opacity", ["1", "0"], [0, SCAN], TD)}
</path>

<line x1="{SPLIT}" y1="18" x2="{SPLIT}" y2="300" stroke="{GRAY}" stroke-opacity="0.3" stroke-dasharray="2 5"/>
<g font-family="{mono}" font-size="12" fill="{GRAY}">
<text x="20" y="290">lidar point cloud — equipment segmentation</text>
<text x="652" y="290">breast dce-mri — tumor segmentation</text>
{legend(20, 312, [(COLORS[k], n) for k, n in (("tank", "tank"), ("pipe", "piping"), ("pump", "pump"), ("structure", "structure"))])}
{legend(652, 312, [(red, "tumor")])}
</g>
</svg>
"""
    with open(path, "w") as f:
        f.write(svg)
    print(f"wrote {path}: {len(svg) / 1024:.0f} KB; tumor in slices {tumor_slices[0] + 1}-{tumor_slices[-1] + 1} "
          f"of {NS}, {int(tumor.sum())} tumor voxels, cycle {T}s")


if __name__ == "__main__":
    main(sys.argv[1])
