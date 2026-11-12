"""Summaries of the mechanistic probes across seeds and architectures (revision)."""
import json, os, sys
import rev_analysis as ra


def load(path):
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path)]
    pr = [r for r in recs if r['type'] == 'probe']
    ab = [r for r in recs if r['type'] == 'ablate']
    for r in ab:
        r.setdefault('mode', 'zero')
    intact = next((r for r in recs if r['type'] == 'intact'), None)
    return pr, ab, intact


def runs():
    out = []
    for M in (2, 16, 2048):
        out.append(('Medium', M, 's0', os.path.join(ra.RES, f'mech_mech_m_M{M}.jsonl')))
        for s in (1, 2):
            out.append(('Medium', M, f's{s}', os.path.join(ra.RES, f'mech2_mech_m_M{M}_s{s}.jsonl')))
        out.append(('Medium', M, 's0r', os.path.join(ra.RES, f'mech2_mech_m_M{M}_s0r.jsonl')))
        out.append(('Small', M, 's0', os.path.join(ra.RES, f'mech2_mech_s_M{M}.jsonl')))
    for M in (16, 2048):
        out.append(('Medium-nc', M, 's0', os.path.join(ra.RES, f'mech_mechnc_m_M{M}.jsonl')))
    return out


def summary():
    rows = []
    for arch, M, seed, path in runs():
        d = load(path)
        if d is None:
            continue
        pr, ab, intact = d
        rec = dict(arch=arch, M=M, seed=seed)
        for mode in ('zero', 'mean'):
            c = sorted([r for r in ab if r['mode'] == mode], key=lambda r: -r['mse_ratio_vs_intact'])
            if c:
                rec[f'{mode}_top'] = (c[0]['layer'], c[0]['head'], c[0]['mse_ratio_vs_intact'], c[0]['omega'])
                rec[f'{mode}_second'] = (c[1]['layer'], c[1]['head'], c[1]['mse_ratio_vs_intact'])
                rec[f'{mode}_layer0_share'] = sum(1 for r in c[:2] if r['layer'] == 0)
            q = sorted([r for r in ab if r['mode'] == mode and 'pool_mse_ratio_vs_intact' in r],
                       key=lambda r: -r['pool_mse_ratio_vs_intact'])
            if q:
                rec[f'{mode}_pool_top'] = (q[0]['layer'], q[0]['head'], q[0]['pool_mse_ratio_vs_intact'],
                                           [r['mse_ratio_vs_intact'] for r in ab if r['mode'] == mode and
                                            r['layer'] == q[0]['layer'] and r['head'] == q[0]['head']][0])
                rec[f'{mode}_pool_second'] = (q[1]['layer'], q[1]['head'], q[1]['pool_mse_ratio_vs_intact'])
        if intact and intact.get('mse_pool') is not None:
            rec['pool_excess_rel_noise'] = (intact['mse_pool'] - intact['mse_pool_dmmse']) / 0.0625
        # routing: largest output-token attention mass among layer-0 heads (final query position);
        # t_route = first checkpoint at which it reaches 0.9; t_half = first checkpoint at which
        # Omega reaches half of its final value
        if pr and len(pr) > 1:
            mx = [(r['step'], max(r['ymass'][0])) for r in pr]
            om_final = pr[-1]['omega']
            rec['t_route'] = next((st for st, y in mx if y >= 0.9), None)
            rec['t_half'] = next((r['step'] for r in pr if om_final > 0 and r['omega'] >= 0.5 * om_final), None)
            rec['route_final'] = mx[-1][1]
            # largest one-checkpoint increase of Omega and where it happens
            jumps = [(pr[i]['omega'] - pr[i - 1]['omega'], pr[i - 1]['step'], pr[i]['step']) for i in range(1, len(pr))]
            rec['max_jump'] = max(jumps)
        rec['omega_final'] = pr[-1]['omega'] if pr else (intact or {}).get('omega')
        rec['wR2_final'] = pr[-1]['wR2'] if pr else None
        rows.append(rec)
        z = rec.get('zero_top'); pz = rec.get('zero_pool_top'); pm = rec.get('mean_pool_top'); mn = rec.get('mean_top')
        print(f"{arch:9s} M={M:5d} {seed:3s} Om={rec['omega_final']:+.3f} fresh: L{z[0]}H{z[1]} x{z[2]:.2f} "
              + (f"(mean x{mn[2]:.2f}) " if mn else '')
              + (f"| pool: L{pz[0]}H{pz[1]} x{pz[2]:.2f} (mean x{pm[2]:.2f}; same head fresh x{pz[3]:.2f}) 2nd x{rec['zero_pool_second'][2]:.2f} " if pz else '')
              + (f"| route>=.9 at {rec.get('t_route')} final {rec.get('route_final'):.2f} half at {rec.get('t_half')} jump {rec['max_jump'][0]:+.2f} in ({rec['max_jump'][1]},{rec['max_jump'][2]}]" if 'max_jump' in rec else ''))
    json.dump(rows, open(os.path.join(ra.RES, 'rev_mech_summary.json'), 'w'), indent=1, default=str)
    return rows


if __name__ == '__main__':
    summary()
