"""Revision analyses: aggregates every run (original and new) into the numbers
reported in the revised manuscript. Usage: python3 rev_analysis.py [section ...]
Sections: table, fine, pred, arch, nl, dsweep, inventory, all."""
import glob, json, math, os, re, random, sys
from collections import defaultdict

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
SIZES = {'s': 'Small', 'm': 'Medium', 'l': 'Large', 'xl': 'XLarge'}


def load(tag):
    p = os.path.join(RES, tag + '.jsonl')
    if not os.path.exists(p):
        return None, []
    lines = [json.loads(l) for l in open(p)]
    metas = [l for l in lines if l.get('type') == 'meta']
    evs = [l for l in lines if l.get('type') == 'eval']
    # resumed runs may repeat eval steps; keep the last record per step
    by = {}
    for e in evs:
        by[e['step']] = e
    evs = [by[s] for s in sorted(by)]
    return (metas[-1] if metas else None), evs


def final(tag, steps=20000):
    meta, evs = load(tag)
    if not evs or evs[-1]['step'] != steps:
        return None
    return evs[-1]


def om(e):
    return e['order_fresh']


def rho(e):
    return e['mse_fresh_model'] / e['mse_fresh_ridge']


def cell_tags(size, M, pool='A'):
    """Tags for a (size, M) cell under pool draw A (base 1000), B (2000), or C (3000)."""
    if pool == 'A':
        tags = [f'{size}_M{M}_s{s}' for s in (0, 1, 2)] + [f'{size}_M{M}']
        return tags
    pre = {'B': 'poolB', 'C': 'poolC'}[pool]
    return [f'{pre}_{size}_M{M}']


def cell_finals(size, M, pool='A'):
    out, seen = [], set()
    for t in cell_tags(size, M, pool):
        e = final(t)
        if e is None:
            continue
        # legacy untagged file is seed 0; avoid double counting with _s0
        key = re.sub(r'_s0$', '', t)
        if key in seen:
            continue
        seen.add(key)
        out.append((t, e))
    return out


def fmt(x, nd=2):
    return f'{x:+.{nd}f}'


# ------------------------------------------------------------------ table 2
GRID = [2, 4, 6, 8, 10, 12, 14, 16, 24, 32, 64, 128, 256, 2048, 0]


def table():
    rows = []
    for size in ('s', 'm', 'l', 'xl'):
        for M in GRID:
            fs = cell_finals(size, M, 'A')
            if not fs:
                continue
            O = [om(e) for _, e in fs]
            R = [rho(e) for _, e in fs]
            rows.append((size, M, len(fs), sum(O) / len(O), min(O), max(O),
                         sum(R) / len(R), min(R), max(R)))
    for r in rows:
        size, M, n, om_, olo, ohi, rm, rlo, rhi = r
        Ms = '∞' if M == 0 else str(M)
        print(f'{SIZES[size]:7s} M={Ms:>5s} n={n} Om={om_:+.3f} [{olo:+.2f},{ohi:+.2f}] '
              f'rho={rm:.2f} [{rlo:.2f},{rhi:.2f}]')
    return rows


# ------------------------------------------------------- fine-resolution boundary
def pool_units(size, M, family_prefix=''):
    """One value per independent pool draw (seeds within a pool are averaged)."""
    units = []
    for pool in ('A', 'B', 'C'):
        fs = cell_finals(size, M, pool)
        if fs:
            units.append(sum(om(e) for _, e in fs) / len(fs))
    return units


def crossing(Ms, vals):
    """Interpolated crossing in log2 M between last negative and first positive mean."""
    means = [sum(v) / len(v) for v in vals]
    for i in range(len(Ms) - 1):
        if means[i] < 0 <= means[i + 1]:
            a, b = math.log2(Ms[i]), math.log2(Ms[i + 1])
            t = -means[i] / (means[i + 1] - means[i])
            return 2 ** (a + t * (b - a)), (Ms[i], Ms[i + 1])
    return None, None


def fine_seeds(size, Ms=(8, 12, 14, 16, 24, 32), B=2000, seed=0):
    """Crossing on pool draw A with seeds as bootstrap units (Large and XLarge)."""
    rng = random.Random(seed)
    xs, units = [], []
    for M in Ms:
        u = [om(e) for _, e in cell_finals(size, M, 'A')]
        if u:
            xs.append(M); units.append(u)
    mstar, iv = crossing(xs, units)
    boots = []
    for _ in range(B):
        m, _ = crossing(xs, [[rng.choice(u) for _ in u] for u in units])
        if m is not None:
            boots.append(m)
    boots.sort()
    lo, hi = boots[int(0.05 * len(boots))], boots[int(0.95 * len(boots)) - 1]
    print(f'{SIZES[size]:7s} pool A, seeds as units: M*={mstar} interval={iv} 90%CI=({lo},{hi})')
    return mstar, iv, lo, hi


def fine(B=2000, seed=0):
    rng = random.Random(seed)
    out = {}
    for size in ('s', 'm', 'l', 'xl'):
        Ms, units = [], []
        for M in (4, 6, 8, 10, 12, 14, 16, 24, 32):
            u = pool_units(size, M)   # one unit per pool draw, seeds within a draw averaged
            if u:
                Ms.append(M); units.append(u)
        if len(Ms) < 2:
            continue
        mstar, iv = crossing(Ms, units)
        boots = []
        for _ in range(B):
            bu = [[rng.choice(u) for _ in u] for u in units]
            m, _ = crossing(Ms, bu)
            if m is not None:
                boots.append(m)
        boots.sort()
        lo = boots[int(0.05 * len(boots))] if boots else None
        hi = boots[int(0.95 * len(boots)) - 1] if boots else None
        print(f'{SIZES[size]:7s} M*={mstar} interval={iv} 90%CI=({lo},{hi}) '
              f'n_boot_valid={len(boots)}/{B}')
        for M, u in zip(Ms, units):
            print(f'    M={M:3d} units={len(u)} mean={sum(u)/len(u):+.3f} vals={[round(x, 3) for x in u]}')
        out[size] = (mstar, iv, lo, hi, Ms, units)
    return out


def fine_per_pool(size, Ms=(4, 6, 8, 10, 12, 14, 16, 24, 32)):
    """Crossing estimated separately for each pool draw (seeds within a draw averaged)."""
    out = {}
    for pool in ('A', 'B', 'C'):
        xs, vals = [], []
        for M in Ms:
            fs = cell_finals(size, M, pool)
            if fs:
                xs.append(M); vals.append([sum(om(e) for _, e in fs) / len(fs)])
        if len(xs) >= 2:
            m, iv = crossing(xs, vals)
            out[pool] = (m, iv, list(zip(xs, [round(v[0], 3) for v in vals])))
            print(f'{SIZES[size]:7s} pool {pool}: M*={m} interval={iv} points={out[pool][2]}')
    return out


# ------------------------------------------------------ prediction-level metrics
def pred(tags=None):
    if tags is None:
        tags = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(RES, '*.jsonl')))
    rows = []
    for t in tags:
        meta, evs = load(t)
        if not evs or 'Dr_fresh' not in evs[-1]:
            continue
        e = evs[-1]
        D = e['Drg_fresh']
        if D <= 0:
            continue
        dr, dg = e['Dr_fresh'] / D, e['Dg_fresh'] / D
        excess = e['mse_fresh_model'] - e['mse_fresh_ridge']
        pool_excess = (e.get('mse_pool_model', float('nan')) - e.get('mse_pool_dmmse', float('nan')))
        rows.append((t, meta['M'], e['step'], om(e), rho(e), dr, dg,
                     e['Dr_fresh'], excess, pool_excess, e.get('Dg_pool', float('nan'))))
        print(f'{t:22s} M={meta["M"]:5d} step={e["step"]:6d} Om={om(e):+.3f} rho={rho(e):6.2f} '
              f'd_r={dr:.3f} d_g={dg:.3f} Dr={e["Dr_fresh"]:.4f} MSEexcess={excess:.4f} '
              f'poolExcess={pool_excess:.4f} Dg_pool={e.get("Dg_pool", float("nan")):.4f}')
    return rows


# ------------------------------------------------------------ architecture control
def arch():
    for name, tags in (('Medium', [f'm_M16_s{s}' for s in (1, 2)] + ['m_M16']),
                       ('Wide', ['wd_M16_s0', 'wd_M16_s1']),
                       ('Deep', ['dp_M16_s0', 'dp_M16_s1']),
                       ('2 heads', ['hd2_M16_s0', 'hd2_M16_s1']),
                       ('8 heads', ['hd8_M16_s0', 'hd8_M16_s1']),
                       ('Large', [f'l_M16_s{s}' for s in (0, 1, 2)]),
                       ('Small', [f's_M16_s{s}' for s in (1, 2)] + ['s_M16']),
                       ('XLarge', [f'xl_M16_s{s}' for s in (0, 1, 2)])):
        fs = [(t, final(t)) for t in tags]
        fs = [(t, e) for t, e in fs if e]
        if not fs:
            print(name, 'pending'); continue
        O = [om(e) for _, e in fs]; R = [rho(e) for _, e in fs]
        print(f'{name:7s} n={len(fs)} Om mean={sum(O)/len(O):+.3f} vals={[round(x,3) for x in O]} '
              f'rho mean={sum(R)/len(R):.2f} vals={[round(x,2) for x in R]}')


# ---------------------------------------------------------------- nonlinear family
def nl():
    for M in (2, 4, 8, 16, 24, 32, 64, 128, 0):
        tags = [f'nl_m_M{M}', f'nl_m_M{M}_s1', f'nl_m_M{M}_s2'] if M else ['nl_m_M0_l1']
        pools = [t for t in (f'nlB_m_M{M}', f'nlC_m_M{M}') if final(t)]
        if pools:
            print(f'    pool replicates at M={M}:', [(t, round(om(final(t)), 3), round(rho(final(t)), 2)) for t in pools])
        fs = [(t, final(t)) for t in tags]
        fs = [(t, e) for t, e in fs if e]
        if not fs:
            print(f'M={M} pending'); continue
        O = [om(e) for _, e in fs]; R = [rho(e) for _, e in fs]
        print(f'nonlinear M={M if M else "inf":>4} n={len(fs)} Om={sum(O)/len(O):+.3f} '
              f'[{min(O):+.2f},{max(O):+.2f}] rho={sum(R)/len(R):.2f} [{min(R):.2f},{max(R):.2f}] '
              f'dmmse/ridge={fs[0][1]["mse_fresh_dmmse"]/fs[0][1]["mse_fresh_ridge"]:.2f}')


def dsweep():
    for d, Ms in ((2, (2, 4, 8, 10, 12, 14, 16, 32)), (6, (4, 8, 16, 18, 20, 24, 28, 32, 64, 128))):
        for M in Ms:
            tags = [f'd{d}_m_M{M}', f'd{d}_m_M{M}_s1', f'd{d}_m_M{M}_s2']
            fs = [(t, final(t)) for t in tags]
            fs = [(t, e) for t, e in fs if e]
            if not fs:
                continue
            O = [om(e) for _, e in fs]
            print(f'd={d} M={M:4d} n={len(fs)} Om={sum(O)/len(O):+.3f} vals={[round(x,3) for x in O]}')


def dsweep_cross():
    """Crossings of the dense family at d = 2, 4, 6 for Medium on pool draw A (seed means where
    several seeds exist, single runs at the fine-grid points), by log2 interpolation."""
    res = {}
    grids = {2: (2, 4, 8, 10, 12, 14, 16, 32), 4: (2, 4, 8, 10, 12, 14, 16, 32),
             6: (4, 8, 16, 18, 20, 24, 28, 32, 64, 128)}
    for d, Ms in grids.items():
        xs, vals = [], []
        for M in Ms:
            if d == 4:
                fs = cell_finals('m', M, 'A'); O = [om(e) for _, e in fs]
            else:
                tags = [f'd{d}_m_M{M}', f'd{d}_m_M{M}_s1', f'd{d}_m_M{M}_s2']
                O = [om(e) for e in (final(t) for t in tags) if e]
            if O:
                xs.append(M); vals.append([sum(O) / len(O)])
        m, iv = crossing(xs, vals)
        res[d] = (m, iv)
        print(f'd={d}: M*={m} interval={iv} alpha=M*/d={m / d if m else None} points={[(x, round(v[0], 3)) for x, v in zip(xs, vals)]}')
    return res


def inventory():
    tags = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(RES, '*.jsonl')))
    n, steps = 0, 0
    for t in tags:
        if t.startswith(('mech_mech', 'mech_mechnc', 'mech2_', 'bench')):
            continue
        meta, evs = load(t)
        if not meta or not evs or evs[-1]['step'] != meta['steps']:
            if meta:
                print('incomplete', t, evs[-1]['step'] if evs else None, meta['steps'])
            continue
        n += 1; steps += meta['steps']
    print('complete training runs:', n, 'steps:', steps)


if __name__ == '__main__':
    secs = sys.argv[1:] or ['all']
    for s in ('table', 'fine', 'pred', 'arch', 'nl', 'dsweep', 'inventory'):
        if 'all' in secs or s in secs:
            print(f'\n===== {s} =====')
            globals()[s]()
            if s == 'fine':
                for size in ('m', 'l', 'xl'):
                    fine_per_pool(size)
                for size in ('l', 'xl'):
                    fine_seeds(size)
