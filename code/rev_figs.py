"""Figures for the revised manuscript, in one shared style.

Style: white background, DejaVu Sans, no in-figure titles, panel letters (A),
(B), ... at the top left, light grid, no top/right spines, 1.8-pt lines,
white-edged markers. Each color has one meaning in every figure:
  architectures   Small blue, Medium orange, Large aqua, XLarge brown;
  task diversity  one ordinal magenta ramp, light (M = 2) to dark (M = 2048);
  task families   dense linear in the Medium orange (dimension shown by line style
                  and marker), sparse indigo, nonlinear black;
  references      ridge and dMMSE as gray dashed lines; zero lines are thin and solid.
The palettes pass the dataviz validator (categorical all-pairs for the architectures,
ordinal checks for the diversity ramp).
Usage: python3 rev_figs.py fig1 fig2 fig3 fig4 fig5"""
import glob, json, math, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rev_analysis as ra

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
os.makedirs(OUT, exist_ok=True)

INK, INK2, MUTED, GRID, BASE = '#111111', '#3d3d3a', '#77766f', '#e4e3dc', '#b9b8ad'
ARCH = {'s': '#2a78d6', 'm': '#eb6834', 'l': '#1baf7a', 'xl': '#8a5a00'}
ARCHNAME = {'s': 'Small', 'm': 'Medium', 'l': 'Large', 'xl': 'XLarge', 'wd': 'Wide', 'dp': 'Deep'}
DIV = {2: '#e393c6', 8: '#da74b7', 12: '#cc56a6', 16: '#b73e92', 32: '#9e2e7d',
       64: '#832367', 128: '#671b51', 2048: '#4d133c'}
DENSE = ARCH['m']                       # dense linear family (all runs are Medium)
DSTYLE = {2: (':', '^'), 4: ('-', 'o'), 6: ('--', 's')}   # input dimension -> line, marker
FAM = {'sparse': '#5343ac', 'nonlinear': '#111111'}
REF = dict(color='#77766f', lw=1.3, ls=(0, (5, 2.5)))   # reference predictors (ridge, dMMSE)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9.5,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.2,
    'axes.edgecolor': BASE, 'axes.linewidth': 0.8, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.labelcolor': INK2, 'figure.dpi': 300,
    'savefig.dpi': 300, 'lines.linewidth': 1.8, 'lines.markersize': 4.8,
    'legend.frameon': False, 'axes.grid': True, 'grid.color': GRID,
    'grid.linewidth': 0.6, 'axes.axisbelow': True, 'axes.spines.top': False,
    'axes.spines.right': False,
})
MK = dict(markeredgecolor='white', markeredgewidth=0.9)


def letter(ax, s):
    ax.text(-0.13, 1.04, f'({s})', transform=ax.transAxes, fontsize=10.5,
            color=INK, va='bottom', ha='left')


def log2x(ax, ticks):
    ax.set_xscale('log', base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def logy(ax, ticks):
    """Logarithmic y axis with plain-number tick labels, shared by every log panel."""
    ax.set_yscale('log')
    ax.set_yticks(ticks, [f'{t:g}' for t in ticks])
    ax.minorticks_off()


def fix_legend(ax):
    """Legend keys: draw markers without the white edge used on data points, so that
    solid and dashed keys are not confused."""
    leg = ax.get_legend()
    if leg is None:
        return
    hs = getattr(leg, 'legend_handles', None) or getattr(leg, 'legendHandles', [])
    for h in hs:
        try:
            if h.get_markeredgecolor() in ('white', '#ffffff', (1.0, 1.0, 1.0, 1.0)) and h.get_markerfacecolor() not in ('white', '#ffffff'):
                h.set_markeredgecolor(h.get_color())
        except Exception:
            pass


def save(fig, name):
    for ax in fig.axes:
        fix_legend(ax)
    fig.savefig(os.path.join(OUT, name), facecolor='white', bbox_inches='tight', pad_inches=0.03)
    print('wrote', name)


# ---------------------------------------------------------------- Figure 1
def fig1():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    (a, b), (c, d) = axs
    grid = [2, 4, 8, 16, 32, 64, 128, 256, 2048]
    for size in ('s', 'm', 'l', 'xl'):
        Ms, mean, lo, hi, rM, rmean, rlo, rhi = [], [], [], [], [], [], [], []
        for M in grid + [0]:
            fs = ra.cell_finals(size, M, 'A')
            if not fs:
                continue
            O = [ra.om(e) for _, e in fs]; R = [ra.rho(e) for _, e in fs]
            if M:
                Ms.append(M); mean.append(sum(O) / len(O)); lo.append(min(O)); hi.append(max(O))
            rM.append(M if M else 16384); rmean.append(sum(R) / len(R)); rlo.append(min(R)); rhi.append(max(R))
        a.plot(Ms, mean, '-o', color=ARCH[size], label=ARCHNAME[size], **MK, zorder=3)
        a.fill_between(Ms, lo, hi, color=ARCH[size], alpha=0.15, lw=0)
        b.plot(rM, rmean, '-o', color=ARCH[size], label=ARCHNAME[size], **MK, zorder=3)
        b.fill_between(rM, rlo, rhi, color=ARCH[size], alpha=0.15, lw=0)
    dm = [(M, ra.cell_finals('m', M, 'A')[0][1]) for M in grid if ra.cell_finals('m', M, 'A')]
    b.plot([M for M, _ in dm], [e['mse_fresh_dmmse'] / e['mse_fresh_ridge'] for _, e in dm],
           label='dMMSE reference', **REF)
    a.axhline(0, color=BASE, lw=0.9, zorder=1)
    a.set_ylim(-1.0, 1.0); log2x(a, [2, 8, 32, 128, 512, 2048])
    a.set_xlabel('task diversity $M$'); a.set_ylabel(r'order parameter $\Omega$')
    a.text(0.03, 0.96, r'$+1$: ridge (ICL)', transform=a.transAxes, color=MUTED, va='top', fontsize=8)
    a.text(0.03, 0.04, r'$-1$: dMMSE (memorization)', transform=a.transAxes, color=MUTED, fontsize=8)
    a.legend(loc='lower right', ncol=1)
    b.axhline(1, color=BASE, lw=1)
    logy(b, [1, 2, 5, 10, 20, 50]); log2x(b, [2, 8, 32, 128, 512, 2048, 16384])
    b.set_xticks([2, 8, 32, 128, 512, 2048, 16384], ['2', '8', '32', '128', '512', '2048', r'$\infty$'])
    b.set_xlabel('task diversity $M$'); b.set_ylabel(r'error ratio $\rho$ (model / ridge)')
    b.legend(loc='upper right', handles=[Line2D([], [], label='dMMSE reference', **REF)])
    letter(a, 'A'); letter(b, 'B')

    # (C) fine-resolution boundary
    fr = ra.fine(B=2000)
    for size in ('s', 'm', 'l', 'xl'):
        if size not in fr:
            continue
        mstar, iv, lo, hi, Ms, units = fr[size]
        sel = [(M, u) for M, u in zip(Ms, units) if 4 <= M <= 32]
        for M, u in sel:
            c.plot([M * (1 + 0.025 * (['s', 'm', 'l', 'xl'].index(size) - 1.5))] * len(u), u, 'o',
                   color=ARCH[size], ms=2.6, alpha=0.45, mew=0)
        c.plot([M for M, _ in sel], [sum(u) / len(u) for _, u in sel], '-o', color=ARCH[size],
               label=ARCHNAME[size], **MK, zorder=3)
        if mstar:
            yoff = -0.58 - 0.075 * ['s', 'm', 'l', 'xl'].index(size)
            if lo and hi and hi > lo:
                c.plot([lo, hi], [yoff, yoff], '-', color=ARCH[size], lw=2.4, alpha=0.75, solid_capstyle='butt')
            c.plot([mstar], [yoff], 'D', color=ARCH[size], ms=4.0, **MK, zorder=4)
    c.axhspan(-0.9, -0.52, color='#f4f3ee', zorder=0, lw=0)
    c.text(4.0, -0.545, r'crossing $M^{*}$ (90% interval)', color=MUTED, fontsize=7.2, va='bottom')
    c.axhline(0, color=BASE, lw=0.9, zorder=1)
    log2x(c, [4, 6, 8, 10, 12, 16, 24, 32]); c.set_xlim(3.7, 34)
    c.set_ylim(-0.9, 0.9); c.set_yticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75])
    c.set_xlabel('task diversity $M$ (fine grid)'); c.set_ylabel(r'order parameter $\Omega$')
    c.legend(loc='upper left')
    letter(c, 'C')

    # (D) prediction-level distances, Medium
    rows = []
    for t in sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(ra.RES, '*.jsonl'))):
        if not re.match(r'^(rr_m_M\d+_s0|mech_m_M\d+_s\d|m_M\d+_s0|poolB_m_M\d+|poolC_m_M\d+|mech_m_M\d+_s0r)$', t):
            continue
        meta, evs = ra.load(t)
        if not evs or evs[-1]['step'] != 20000 or 'Dr_fresh' not in evs[-1] or meta['layers'] != 3:
            continue
        e = evs[-1]
        rows.append((meta['M'], e['Dr_fresh'] / e['Drg_fresh'], e['Dg_fresh'] / e['Drg_fresh']))
    if rows:
        byM = {}
        for M, dr, dg in rows:
            byM.setdefault(M, []).append((dr, dg))
        Ms = sorted(byM)
        mr = [sum(v[0] for v in byM[M]) / len(byM[M]) for M in Ms]
        mg = [sum(v[1] for v in byM[M]) / len(byM[M]) for M in Ms]
        for M, dr, dg in rows:
            d.plot([M], [dr], 'o', color=ARCH['m'], ms=2.6, alpha=0.4, mew=0)
            d.plot([M], [dg], 's', color=ARCH['m'], ms=2.4, alpha=0.4, mew=0)
        d.plot(Ms, mr, '-o', color=ARCH['m'], label=r'$\delta_r$ (distance to ridge)', **MK, zorder=3)
        d.plot(Ms, mg, '--s', color=ARCH['m'], mfc='white', mec=ARCH['m'], mew=1.0, ms=4.2,
               label=r'$\delta_g$ (distance to dMMSE)', zorder=3)
        logy(d, [0.01, 0.03, 0.1, 0.3, 1]); log2x(d, [2, 8, 32, 128, 512, 2048]); d.set_ylim(0.008, 2.5)
        d.set_xlabel('task diversity $M$ (Medium)'); d.set_ylabel('normalized squared distance')
        d.legend(loc='lower left')
    letter(d, 'D')
    fig.tight_layout(w_pad=2.2, h_pad=1.6)
    save(fig, 'fig1_phase_diagram.png')


# ---------------------------------------------------------------- Figure 3
def fig3():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.4, 2.75), gridspec_kw={'width_ratios': [1, 1.1, 1.05]})
    dy = {128: 0.02, 2048: -0.1, 8: 0.04, 2: -0.08}
    for M in (2, 8, 16, 32, 128, 2048):
        _, evs = ra.load(f'l_M{M}_s0')
        a.plot([e['step'] / 1000 for e in evs], [ra.om(e) for e in evs], '-', color=DIV[M], lw=1.7)
        a.annotate(f'$M={M}$', (20, ra.om(evs[-1]) + dy.get(M, 0)), xytext=(4, 0),
                   textcoords='offset points', color=DIV[M], fontsize=7.6, va='center')
    a.axhline(0, color=BASE, lw=0.9, zorder=1)
    a.set_ylim(-1, 1); a.set_xlim(0, 27.5); a.set_xticks([0, 5, 10, 15, 20])
    a.set_xlabel('training step (thousands)'); a.set_ylabel(r'order parameter $\Omega$')
    letter(a, 'A')
    runs = {}
    for p in sorted(glob.glob(os.path.join(ra.RES, '*_long.jsonl'))):
        m = re.match(r'([sml])_M(\d+)_s(\d)_long$', os.path.basename(p)[:-6])
        if not m:
            continue
        _, evs = ra.load(os.path.basename(p)[:-6])
        if evs and evs[-1]['step'] == 100000:
            runs[(m.group(1), int(m.group(2)), int(m.group(3)))] = evs
    for (size, M, seed), evs in sorted(runs.items()):
        if size != 'm':
            continue
        b.plot([e['step'] / 1000 for e in evs], [ra.om(e) for e in evs], '-', color=DIV[M],
               lw=1.8 if seed == 0 else 1.0, alpha=1 if seed == 0 else 0.6)
        if seed == 0:
            b.annotate(f'$M={M}$', (100, ra.om(evs[-1])), xytext=(4, 0), textcoords='offset points',
                       color=DIV[M], fontsize=8, va='center')
    b.axhline(0, color=BASE, lw=0.9, zorder=1)
    b.set_ylim(-1, 1); b.set_xlim(0, 118); b.set_xlabel('training step (thousands)')
    b.set_ylabel(r'order parameter $\Omega$')
    letter(b, 'B')
    cells = sorted(runs, key=lambda k: (['s', 'm', 'l'].index(k[0]), k[1], k[2]))
    labels = []
    for i, (size, M, seed) in enumerate(cells):
        evs = runs[(size, M, seed)]
        o20 = next(ra.om(e) for e in evs if e['step'] == 20000)
        o100 = ra.om(evs[-1])
        c.plot([o20, o100], [i, i], '-', color=ARCH[size], lw=1.4, alpha=0.6)
        c.plot([o20], [i], 'o', color='white', mec=ARCH[size], mew=1.3, ms=5)
        c.plot([o100], [i], 'o', color=ARCH[size], ms=5.4, **MK)
        labels.append(f'{ARCHNAME[size]} $M$={M} s{seed}')
    c.axvline(0, color=BASE, lw=0.9)
    c.set_yticks(range(len(labels)), labels, fontsize=7.2)
    c.set_xlim(-0.6, 1.0); c.set_xlabel(r'$\Omega$ at 20k (open) and 100k (filled)')
    c.grid(axis='y', visible=False)
    letter(c, 'C')
    fig.tight_layout(w_pad=1.4)
    save(fig, 'fig3_dynamics_stability.png')



# ---------------------------------------------------------------- Figure 4
def load_mech(tag, v2=True):
    """Probe trajectory and ablation records; v2 files carry zero and mean ablation."""
    p = os.path.join(ra.RES, (f'mech2_{tag}.jsonl' if v2 else f'mech_{tag}.jsonl'))
    if not os.path.exists(p):
        return None
    recs = [json.loads(l) for l in open(p)]
    pr = [r for r in recs if r['type'] == 'probe']
    ab = [r for r in recs if r['type'] == 'ablate']
    for r in ab:
        r.setdefault('mode', 'zero')
    return pr, ab


def mech_runs(M, arch='m'):
    """(label, probes, ablations, is_first_seed) for the curriculum runs at diversity M.
    The first seed's 500-step trajectory comes from the original instrumented run (zero-ablation
    only); its final-checkpoint ablations of both types come from a rerun with saved weights."""
    out = []
    if arch == 'm':
        o = load_mech(f'mech_m_M{M}', v2=False)
        if o:
            out.append(('s0', o[0], o[1], True))
        for s in (1, 2):
            o = load_mech(f'mech_m_M{M}_s{s}')
            if o:
                out.append((f's{s}', o[0], o[1], False))
    else:
        o = load_mech(f'mech_s_M{M}')
        if o:
            out.append(('s0', o[0], o[1], True))
    return out


def ablation_runs(M):
    """Runs with final-checkpoint zero- and mean-ablation (rerun of seed 0, seeds 1 and 2)."""
    out = []
    for tag in (f'mech_m_M{M}_s0r', f'mech_m_M{M}_s1', f'mech_m_M{M}_s2'):
        o = load_mech(tag)
        if o:
            out.append((tag, o[1]))
    return out


def top_head(ab, layer=0, mode='zero'):
    c = [r for r in ab if r['layer'] == layer and r['mode'] == mode]
    return max(c, key=lambda r: r['mse_ratio_vs_intact'])['head'] if c else None


def fig4():
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.3))
    (a, b), (c, d) = axs
    for M in (2, 16, 2048):
        for lab, pr, ab, first in mech_runs(M):
            a.plot([r['step'] / 1000 for r in pr], [r['omega'] for r in pr], '-', color=DIV[M],
                   lw=1.7 if first else 0.9, alpha=1 if first else 0.75)
        nc = load_mech(f'mechnc_m_M{M}', v2=False) if M != 2 else None
        if nc:
            a.plot([r['step'] / 1000 for r in nc[0]], [r['omega'] for r in nc[0]], ':', color=DIV[M], lw=1.5)
        runs = mech_runs(M)
        if runs:
            a.annotate(f'$M={M}$', (20, runs[0][1][-1]['omega']), xytext=(4, 0), textcoords='offset points',
                       color=DIV[M], fontsize=7.8, va='center')
    a.axhline(0, color=BASE, lw=0.9, zorder=1)
    a.set_xlim(0, 25.5); a.set_xticks([0, 5, 10, 15, 20]); a.set_ylim(-0.8, 0.9)
    a.set_xlabel('training step (thousands)'); a.set_ylabel(r'order parameter $\Omega$')
    a.legend(handles=[Line2D([], [], color=MUTED, lw=1.7, label='seed 0'),
                      Line2D([], [], color=MUTED, lw=0.9, label='seeds 1, 2'),
                      Line2D([], [], color=MUTED, lw=1.5, ls=':', label='no curriculum')],
             loc='center', bbox_to_anchor=(0.66, 0.36), fontsize=7.4, handlelength=2.2)
    letter(a, 'A')
    for M in (2048, 16, 2):
        for lab, pr, ab, first in mech_runs(M):
            b.plot([r['step'] / 1000 for r in pr], [max(r['ymass'][0]) for r in pr], '-', color=DIV[M],
                   lw=1.5 if first else 0.9, alpha=1 if first else 0.75)
    b.axvline(2.667, color=BASE, lw=0.9, ls=':'); b.axvline(5.334, color=BASE, lw=0.9, ls=':')
    b.text(2.667, 0.03, ' $d_{\\mathrm{active}}=3$', color=MUTED, fontsize=7, rotation=90, va='bottom')
    b.text(5.334, 0.03, ' $d_{\\mathrm{active}}=4$', color=MUTED, fontsize=7, rotation=90, va='bottom')
    b.set_xlim(0, 20.5); b.set_xticks([0, 5, 10, 15, 20])
    b.set_ylim(0, 1.05); b.set_xlabel('training step (thousands)')
    b.set_ylabel('max. output-token attention')
    b.legend(handles=[Line2D([], [], color=DIV[M], lw=1.5, label=f'$M={M}$') for M in (2048, 16, 2)],
             loc='lower right', fontsize=7.4, handlelength=1.8)
    letter(b, 'B')
    for ax, key in ((c, 'mse_ratio_vs_intact'), (d, 'pool_mse_ratio_vs_intact')):
        for M in (2, 16, 2048):
            for tag, ab in ablation_runs(M):
                for mode in ('zero', 'mean'):
                    vals = sorted((r[key] for r in ab if r['mode'] == mode and key in r), reverse=True)
                    if not vals:
                        continue
                    kw = dict(color=DIV[M], ms=3.4, lw=1.1, alpha=0.9)
                    if mode == 'zero':
                        ax.plot(range(1, len(vals) + 1), vals, '-o', mec='white', mew=0.6, **kw)
                    else:
                        ax.plot(range(1, len(vals) + 1), vals, '--o', mfc='white', mec=DIV[M], mew=0.9, **kw)
        logy(ax, [1, 2, 5, 10, 20, 50]); ax.axhline(1, color=BASE, lw=1)
        ax.set_xticks([1, 3, 6, 9, 12]); ax.set_xlabel('head rank (by ablation impact)')
    c.set_ylabel('relative fresh-task error')
    d.set_ylabel('relative pool-context error')
    c.set_ylim(0.9, 60); d.set_ylim(0.9, 60)
    c.legend(handles=[Line2D([], [], color=DIV[M], lw=1.4, label=f'$M={M}$') for M in (2048, 16, 2)] +
             [Line2D([], [], color=MUTED, lw=1.2, marker='o', ms=3.6, mec=MUTED, label='zero-ablation'),
              Line2D([], [], color=MUTED, lw=1.0, ls='--', marker='o', ms=3.6, mfc='white', label='mean-ablation')],
             loc='upper right', fontsize=7.2, handlelength=2.6)
    letter(c, 'C'); letter(d, 'D')
    fig.tight_layout(w_pad=2.0, h_pad=1.5)
    save(fig, 'fig4_mechanism.png')


# ---------------------------------------------------------------- Figure 5
def family_curve(prefix_fn, Ms):
    xs, means, pts = [], [], []
    for M in Ms:
        vals = [ra.om(e) for e in (ra.final(t) for t in prefix_fn(M)) if e]
        if vals:
            xs.append(M); means.append(sum(vals) / len(vals)); pts.append(vals)
    return xs, means, pts


DENSE_RUNS = {
    2: lambda M: [f'd2_m_M{M}', f'd2_m_M{M}_s1', f'd2_m_M{M}_s2'],
    4: lambda M: [f'm_M{M}', f'm_M{M}_s0', f'm_M{M}_s1', f'm_M{M}_s2'],
    6: lambda M: [f'd6_m_M{M}', f'd6_m_M{M}_s1', f'd6_m_M{M}_s2'],
}


def dense_line(ax, d, Ms, **kw):
    ls, mk = DSTYLE[d]
    xs, means, _ = family_curve(DENSE_RUNS[d], Ms)
    ax.plot(xs, means, ls=ls, marker=mk, color=DENSE, ms=3.6, **MK, label=f'dense, $d={d}$', zorder=3, **kw)


def fig5():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3))
    a.axvspan(4, 16, color=FAM['sparse'], alpha=0.07, zorder=0)
    a.text(8, 0.93, 'preregistered\nsparse interval', color=INK2, fontsize=7, ha='center', va='top')
    for d in (2, 4, 6):
        dense_line(a, d, [2, 4, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 64, 128])
    xs, means, pts = family_curve(lambda M: [f'sp_m_M{M}_s0', f'sp_m_M{M}_s1'], [2, 4, 8, 16, 32, 128])
    a.plot(xs, means, '-o', color=FAM['sparse'], ms=3.8, **MK, label='sparse (held out)', zorder=4)
    xs, means, pts = family_curve(lambda M: [f'spm_m_M{M}_s0', f'spm_m_M{M}_s1'], [4, 8, 16, 32, 64])
    a.plot(xs, means, '--o', color=FAM['sparse'], ms=3.6, mfc='white', mec=FAM['sparse'], mew=1.0,
           label='sparse, signal-matched', zorder=4)
    a.axhline(0, color=BASE, lw=0.9, zorder=1)
    log2x(a, [2, 4, 8, 16, 32, 64, 128]); a.set_ylim(-1.0, 1.0)
    a.set_xlabel('task diversity $M$ (Medium)'); a.set_ylabel(r'order parameter $\Omega$')
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.0, handlelength=2.4,
             columnspacing=1.0)
    letter(a, 'A')
    xs, means, pts = family_curve(lambda M: [f'nl_m_M{M}', f'nl_m_M{M}_s1', f'nl_m_M{M}_s2'],
                                  [2, 4, 8, 16, 24, 32, 64, 128])
    for M, v in zip(xs, pts):
        b.plot([M] * len(v), v, 'o', color=FAM['nonlinear'], ms=2.6, alpha=0.35, mew=0)
    for t in ('nlB_m_M16', 'nlC_m_M16'):
        e = ra.final(t)
        if e:
            b.plot([16 * 1.06], [ra.om(e)], 'o', mfc='white', mec=FAM['nonlinear'], mew=0.9, ms=3.6, zorder=5)
    b.plot(xs, means, '-o', color=FAM['nonlinear'], ms=3.8, **MK, label='nonlinear (random features)', zorder=4)
    dense_line(b, 4, [2, 4, 8, 10, 12, 14, 16, 32, 64, 128])
    b.axhline(0, color=BASE, lw=0.9, zorder=1)
    log2x(b, [2, 4, 8, 16, 32, 64, 128]); b.set_ylim(-1.0, 1.0)
    b.set_xlabel('task diversity $M$ (Medium)'); b.set_ylabel(r'order parameter $\Omega$')
    b.legend(handles=[Line2D([], [], color=FAM['nonlinear'], marker='o', ms=3.8, mec='white', label='nonlinear (random features)'),
                      Line2D([], [], color=FAM['nonlinear'], lw=0, marker='o', ms=3.6, mfc='white', mec=FAM['nonlinear'], label='nonlinear, other pools'),
                      Line2D([], [], color=DENSE, marker='o', ms=3.6, mec='white', label='dense, $d=4$')],
             loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7.0, handlelength=2.4,
             columnspacing=1.0)
    letter(b, 'B')
    fig.tight_layout(w_pad=2.2)
    save(fig, 'fig5_families.png')


# ---------------------------------------------------------------- Figure 2
def fig2():
    """Demonstration diversity at inference: error against the rank of the demonstration
    inputs (A) and the reduction from rank one to rank four against pretraining diversity (B)."""
    p = os.path.join(ra.RES, 'rev_demo.json')
    res = [r for r in json.load(open(p)) if r['kind'] == 'rank']
    byM, ridge = {}, {}
    for r in res:
        byM.setdefault(r['M'], {}).setdefault(r['level'], []).append(r['mse'])
        ridge[r['level']] = r['mse_ridge']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1, 1]})
    for M in sorted(byM):
        lv = sorted(byM[M])
        a.plot(lv, [sum(byM[M][l]) / len(byM[M][l]) for l in lv], '-o', color=DIV[M], ms=3.8, **MK,
               label=f'$M={M}$', zorder=3)
    lv = sorted(ridge)
    a.plot(lv, [ridge[l] for l in lv], label='ridge (Bayes)', zorder=4, **REF)
    logy(a, [0.1, 0.2, 0.5, 1, 2, 5]); a.set_xticks([1, 2, 3, 4])
    a.set_ylim(0.085, 8.5)
    a.set_xlabel('rank of demonstration inputs'); a.set_ylabel('fresh-task error (MSE)')
    a.legend(loc='lower left', ncol=2, fontsize=7.0, handlelength=2.0, columnspacing=0.9)
    letter(a, 'A')
    Ms = sorted(byM)
    red = [(sum(byM[M][1]) / len(byM[M][1])) / (sum(byM[M][4]) / len(byM[M][4])) for M in Ms]
    b.plot(Ms, red, '-', color=INK2, lw=1.0, zorder=2)
    for M, v in zip(Ms, red):
        b.plot([M], [v], 'o', color=DIV[M], ms=5.6, **MK, zorder=3)
    b.axhline(ridge[1] / ridge[4], label='ridge (Bayes)', **REF)
    logy(b, [1, 2, 5, 10, 20, 50]); log2x(b, [2, 8, 32, 128, 512, 2048])
    b.set_ylim(0.8, 60)
    b.set_xlabel('task diversity $M$ in pretraining')
    b.set_ylabel('error reduction, rank 1 to rank 4')
    b.legend(loc='upper left', handles=[Line2D([], [], label='ridge (Bayes)', **REF)])
    letter(b, 'B')
    fig.tight_layout(w_pad=2.4)
    save(fig, 'fig2_demonstrations.png')

# ---------------------------------------------------------------- Figure 6: pretrained checkpoints
PYTHIA = {'70m': 7.0e7, '160m': 1.6e8, '410m': 4.1e8, '1b': 1.0e9, '1.4b': 1.4e9, '2.8b': 2.8e9}
PYCOL = ['#5bb6cf', '#3a9fc0', '#2486a8', '#166c8c', '#0e526d', '#083a4f']   # one validated hue ramp, light to dark


def probe_results():
    """Per-prompt results of the probe of Pythia checkpoints (code/llm_probe/results)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_probe'))
    import llm_probe as lp
    res = lp.load_results(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llm_probe', 'results'))
    return lp, res


def fig6():
    lp, res = probe_results()
    sizes = [z for z in PYTHIA if any(m == z for m, _, _ in res)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw=dict(width_ratios=[1.45, 1]))
    any_rows = next(iter(res.values()))
    s0 = lp.summarize_rows(any_rows)
    base_ctx, base_nn = s0['mse_ctxmean'] / s0['mse_bayes'], s0['mse_nn1'] / s0['mse_bayes']
    for i, z in enumerate(sizes):
        steps = sorted(st for m, st, k in res if m == z and k == 10)
        rho = [lp.summarize_rows(res[(z, st, 10)])['rho'] for st in steps]
        a.plot(steps, rho, '-o', color=PYCOL[i], ms=3.6, lw=1.5, label=f'Pythia-{z.upper().replace("M", "M")}', **MK)
    for ax in (a, b):
        ax.axhline(base_ctx, **REF); ax.axhline(base_nn, color=MUTED, lw=1.1, ls=(0, (1, 2)))
        ax.axhline(1, color=BASE, lw=0.9)
    a.text(1.02, base_ctx, 'average of\ndemonstrated outputs', transform=a.get_yaxis_transform(),
           fontsize=7, color=MUTED, va='center')
    a.text(1.02, base_nn, 'output of nearest\ndemonstration', transform=a.get_yaxis_transform(),
           fontsize=7, color=MUTED, va='center')
    a.text(1.02, 1, 'Bayes', transform=a.get_yaxis_transform(), fontsize=7, color=MUTED, va='center')
    a.set_xscale('log'); a.set_xticks([512, 1000, 4000, 16000, 64000, 143000],
                                      ['512', '1k', '4k', '16k', '64k', '143k']); a.minorticks_off()
    logy(a, [1, 3, 10, 30, 100, 300, 1000])
    a.set_xlabel('pretraining step'); a.set_ylabel('error ratio to the Bayes predictor')
    a.legend(loc='lower left', fontsize=7.6)
    # (B) final checkpoint against size, with 95% bootstrap intervals over prompts
    xs, ys, lo, hi = [], [], [], []
    for i, z in enumerate(sizes):
        last = max(st for m, st, k in res if m == z and k == 10)
        sm = lp.summarize_rows(res[(z, last, 10)], boot=1000)
        xs.append(PYTHIA[z]); ys.append(sm['rho']); lo.append(sm['rho_lo']); hi.append(sm['rho_hi'])
        b.errorbar([PYTHIA[z]], [sm['rho']], yerr=[[sm['rho'] - sm['rho_lo']], [sm['rho_hi'] - sm['rho']]],
                   fmt='o', color=PYCOL[i], ms=4.8, capsize=0, elinewidth=1.4, mec='white', mew=0.9)
    b.plot(xs, ys, '-', color=MUTED, lw=0.9, zorder=1)
    b.set_xscale('log'); b.set_xticks([PYTHIA[z] for z in sizes], [z.upper() for z in sizes]); b.minorticks_off()
    logy(b, [1, 3, 10, 30, 100, 300])
    b.set_xlabel('model size (parameters)'); b.set_ylabel('error ratio, final checkpoint')
    letter(a, 'A'); letter(b, 'B')
    fig.tight_layout(w_pad=2.2)
    save(fig, 'fig6_pretrained.png')


if __name__ == '__main__':
    for name in sys.argv[1:]:
        globals()[name]()
