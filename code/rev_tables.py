"""LaTeX tables for the revised manuscript (Table 2 and the appendix inventory)."""
import glob, json, os, re, sys
import rev_analysis as ra

GRID = {'s': [2, 4, 6, 8, 10, 12, 16, 32, 64, 128, 256, 2048, 0],
        'm': [2, 4, 8, 10, 12, 14, 16, 32, 64, 128, 256, 2048, 0],
        'l': [2, 4, 8, 10, 12, 14, 16, 24, 32, 64, 128, 256, 2048, 0],
        'xl': [2, 8, 12, 16, 24, 32, 128]}


def fr(x):
    return f'{x:.1f}' if x >= 10 else f'{x:.2f}'


def table2():
    lines = [r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcllll@{}}', r'\toprule',
             r'\textbf{Architecture} & \textbf{\boldmath$M$} & \textbf{\boldmath$n$} & '
             r'\textbf{\boldmath$\Omega$ mean} & \textbf{\boldmath$\Omega$ range} & '
             r'\textbf{\boldmath$\rho$ mean} & \textbf{\boldmath$\rho$ range} \\', r'\midrule']
    first = True
    for size in ('s', 'm', 'l', 'xl'):
        rows = []
        for M in GRID[size]:
            fs = ra.cell_finals(size, M, 'A')
            if not fs:
                continue
            O = [ra.om(e) for _, e in fs]; R = [ra.rho(e) for _, e in fs]
            n = len(fs)
            Ms = r'$\infty$' if M == 0 else str(M)
            om_m = 'n/a' if M == 0 else f'${sum(O)/n:+.3f}$'
            om_r = '--' if (n == 1 or M == 0) else f'$[{min(O):+.2f}, {max(O):+.2f}]$'
            r_m = fr(sum(R) / n)
            r_r = '--' if n == 1 else f'$[{fr(min(R))}, {fr(max(R))}]$'
            rows.append(f'{ra.SIZES[size]} & {Ms} & {n} & {om_m} & {om_r} & {r_m} & {r_r} \\\\')
        if not first:
            lines.append(r'\midrule')
        first = False
        lines += rows
    lines += [r'\bottomrule', r'\end{tabular*}']
    return '\n'.join(lines)


def count(pattern, steps=20000):
    n = 0
    for p in glob.glob(os.path.join(ra.RES, pattern + '.jsonl')):
        t = os.path.basename(p)[:-6]
        meta, evs = ra.load(t)
        if meta and evs and evs[-1]['step'] == meta['steps']:
            n += 1
    return n


def done(tags):
    return sum(1 for t in tags if ra.final(t, 100000 if t.endswith('_long') else 20000))


def inventory():
    rows = []
    def add(name, arch, fam, Ms, reps, steps, runs):
        rows.append((name, arch, fam, Ms, reps, steps, runs))
    ph_s = done([f's_M{M}_s{s}' for M in (4, 8, 16, 32, 64, 128, 256, 2048) for s in (1, 2)] +
                [f's_M{M}' for M in (2, 16, 128, 2048, 0)] + [f's_M{M}_s0' for M in (4, 8, 32, 64, 256)] +
                [f's_M2_s{s}' for s in (1, 2)])
    ph_m = done([f'm_M{M}_s{s}' for M in (4, 8, 16, 32, 64, 128, 256, 2048) for s in (1, 2)] +
                [f'm_M{M}' for M in (2, 16, 128, 2048, 0)] + [f'm_M{M}_s0' for M in (4, 8, 32, 64, 256)] +
                [f'm_M2_s{s}' for s in (1, 2)])
    ph_l = done([f'l_M{M}_s{s}' for M in (2, 4, 8, 16, 32, 64, 128, 256, 2048, 0) for s in (0, 1, 2)])
    ph_xl = done([f'xl_M{M}_s0' for M in (2, 8, 12, 16, 24, 32, 128)] + [f'xl_M16_s{s}' for s in (1, 2)])
    add('Phase diagram', 'Small, Medium', 'dense, 4', r'2--2048, $\infty$', r'3 seeds ($\infty$: 1)', '20k', ph_s + ph_m)
    add('Phase diagram', 'Large', 'dense, 4', r'2--2048, $\infty$', r'3 seeds at 6 values; 1 elsewhere', '20k', ph_l)
    add('Capacity extension', 'XLarge', 'dense, 4', '2, 8, 12, 16, 24, 32, 128', '3 seeds at 16; 1 elsewhere', '20k', ph_xl)
    fine_m = done([f'{p}_m_M{M}' for p in ('poolB', 'poolC') for M in (10, 12, 14)] + [f'm_M{M}_s0' for M in (10, 12, 14)])
    fine_s = done([f's_M{M}_s0' for M in (6, 10, 12)] + [f'poolB_s_M{M}' for M in (6, 8, 10, 12, 16)])
    fine_l = done([f'l_M{M}_s{s}' for M in (12, 14) for s in (0, 1, 2)] + ['l_M24_s0'])
    add('Fine resolution', 'Small, Medium, Large', 'dense, 4', '6--14 (and 8, 16); Large: 12, 14, 24', 'pools A--C (Medium), A--B (Small); 3 seeds at 12, 14 (Large)', '20k', fine_m + fine_s + fine_l)
    add('Pool replication', 'Medium', 'dense, 4', '2, 8, 16, 32, 128', 'pool B; pool C at 8, 16, 32', '20k',
        done([f'poolB_m_M{M}' for M in (2, 8, 16, 32, 128)] + [f'poolC_m_M{M}' for M in (8, 16, 32)]))
    add('Pool replication', 'Large, XLarge', 'dense, 4', '12, 14, 16, 24 (Large); 16, 24 (XLarge)', 'pool B, 1 seed', '20k',
        done([f'poolB_l_M{M}' for M in (12, 14, 16, 24)] + [f'poolB_xl_M{M}' for M in (16, 24)]))
    longs = [os.path.basename(p)[:-6] for p in glob.glob(os.path.join(ra.RES, '*_long.jsonl'))]
    add('Extended training', 'Small, Medium, Large', 'dense, 4', '8, 16, 32, 64', '1--2 seeds; 2 learning rates', '100k', done(longs))
    mech = done(['mech_m_M2', 'mech_m_M16', 'mech_m_M2048'] + [f'mech_m_M{M}_s{s}' for M in (2, 16, 2048) for s in (1, 2)])
    mech_s = done([f'mech_s_M{M}' for M in (2, 16, 2048)])
    mech_r = done([f'mech_m_M{M}_s0r' for M in (2, 16, 2048)])
    add('Instrumented', 'Medium, Small', 'dense, 4', '2, 16, 2048', '3 seeds (Medium), 1 (Small)', '20k', mech + mech_s)
    add('Curriculum-free', 'Medium', 'dense, 4', r'16, 2048, $\infty$', '1 seed', '20k', done(['mechnc_m_M16', 'mechnc_m_M2048', 'nc_m_M0']))
    add('Architecture control', 'Wide, Deep, 2 and 8 heads', 'dense, 4', '16', '2 seeds', '20k', done(['wd_M16_s0', 'wd_M16_s1', 'dp_M16_s0', 'dp_M16_s1', 'hd2_M16_s0', 'hd2_M16_s1', 'hd8_M16_s0', 'hd8_M16_s1']))
    d2 = [os.path.basename(p)[:-6] for p in glob.glob(os.path.join(ra.RES, 'd2_m_M*.jsonl'))]
    d6 = [os.path.basename(p)[:-6] for p in glob.glob(os.path.join(ra.RES, 'd6_m_M*.jsonl'))]
    add('Dimension sweep', 'Medium', 'dense, 2 and 6 (no curriculum at 2)', '2--32 and 4--128, finer near the crossings', '3 seeds at crossing cells', '20k', done(d2 + d6))
    sp = [os.path.basename(p)[:-6] for p in glob.glob(os.path.join(ra.RES, 'sp_m_M*.jsonl')) if not p.endswith('s0r.jsonl')]
    add('Sparse (preregistered)', 'Medium', r'sparse, $\tau_s = 1$', '2, 4, 8, 16, 32, 128', '2 seeds', '20k', done(sp))
    spm = [os.path.basename(p)[:-6] for p in glob.glob(os.path.join(ra.RES, 'spm_m_M*.jsonl'))]
    add('Sparse, signal-matched', 'Medium', r'sparse, $\tau_s = \sqrt{2}$', '4, 8, 16, 32, 64', '2 seeds', '20k', done(spm))
    nl = [os.path.basename(p)[:-6] for p in glob.glob(os.path.join(ra.RES, 'nl*_m_M*.jsonl'))]
    add('Nonlinear', 'Medium', 'random features, 4', r'2--128, $\infty$', '3 seeds at 16, 32; pools B, C at 16; 1 elsewhere', '20k', done(nl))
    add('Reruns with saved weights', 'Medium', 'dense, 4; sparse', '2, 8, 16, 32, 128, 2048', '1 seed', '20k',
        done([f'rr_m_M{M}_s0' for M in (8, 32, 128)] + [f'mech_m_M{M}_s0r' for M in (2, 16, 2048)] +
             [f'sp_m_M{M}_s0r' for M in (32, 128)]))
    total = sum(r[-1] for r in rows)
    lines = [r'\setlength{\tabcolsep}{4pt}',
             r'\begin{tabularx}{\textwidth}{@{}L{0.2\textwidth}L{0.165\textwidth}L{0.13\textwidth}YYr@{}}', r'\toprule',
             r'\textbf{Campaign} & \textbf{Architecture} & \textbf{Family, $d$} & \textbf{Diversity $M$} & '
             r'\textbf{Replication} & \textbf{Runs} \\', r'\midrule']
    for r in rows:
        name, arch, fam, Ms, reps, steps, runs = r
        if steps != '20k':
            name = f'{name} ({steps} steps)'
        lines.append(' & '.join(str(x) for x in (name, arch, fam, Ms, reps, runs)) + r' \\')
    lines += [r'\midrule', f'Total & & & & & {total} \\\\', r'\bottomrule', r'\end{tabularx}']
    return '\n'.join(lines), total


if __name__ == '__main__':
    print(table2())
    t, n = inventory()
    print(t)
    print('TOTAL', n)
