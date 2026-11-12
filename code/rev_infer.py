"""Inference-only analyses on saved final models (revision).

1. boot: bootstrap standard errors of Omega and rho over the 512 shared
   evaluation contexts for representative final models.
2. demo: inference-time demonstration diversity. For networks pretrained at
   different diversities M, the 11 demonstrations of each fresh-task context are
   restricted to a random r-dimensional subspace of the input space (r = 1..4),
   while the query stays full-rank; a second manipulation rescales all inputs by
   a factor s (covariate shift). Ridge and dMMSE are recomputed on the
   manipulated contexts, so both references remain exact.
Usage: python3 rev_infer.py boot|demo"""
import json, math, os, sys
import torch
import icl_pilot as lib

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
ARCH = {'m': (3, 64, 4), 's': (2, 32, 2), 'l': (4, 128, 4)}


def setup(family='linear', d=4):
    lib.D, lib.K = d, 12
    lib.LAM = (lib.SIGMA / lib.TAU) ** 2
    lib.FAMILY = family
    lib.NL_P, lib.NL_SCALE, lib.NL_TAU = 4, 1.0, 2.0
    if family == 'nonlinear':
        lib.init_features()


def load_model(path, arch='m'):
    L, dm, h = ARCH[arch]
    m = lib.ICLTransformer(L, dm, h)
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


def model_path(tag):
    p = os.path.join(RES, f'final_{tag}.pt')
    if os.path.exists(p):
        return p
    import glob
    c = sorted(glob.glob(os.path.join(RES, f'ckpt_{tag}_20000.pt')))
    return c[0] if c else None


@torch.no_grad()
def stats(model, xs, ys, pool):
    pm = model(xs, ys)[:, -1]
    y = ys[:, -1]
    r = lib.ridge_pred(xs, ys)
    g = lib.dmmse_pred(xs, ys, pool) if pool is not None else r
    return pm, y, r, g


def summarize(pm, y, r, g):
    Dr = ((pm - r) ** 2).mean().item(); Dg = ((pm - g) ** 2).mean().item()
    D = ((r - g) ** 2).mean().item()
    mse = ((pm - y) ** 2).mean().item(); mr = ((r - y) ** 2).mean().item()
    mg = ((g - y) ** 2).mean().item()
    return dict(omega=(Dg - Dr) / (Dg + Dr + 1e-12), rho=mse / mr, dmmse_rho=mg / mr,
                d_r=Dr / D if D > 0 else float('nan'), d_g=Dg / D if D > 0 else float('nan'),
                mse=mse, mse_ridge=mr, Dr=Dr)


def boot():
    setup()
    gf = torch.Generator().manual_seed(777)
    fx, fy = lib.make_batch(None, 512, gf)
    out = {}
    for tag, M, base in [('mech_m_M2_s1', 2, 1000), ('m_M12_s0', 12, 1000), ('mech_m_M16_s1', 16, 1000),
                         ('rr_m_M32_s0', 32, 1000), ('rr_m_M128_s0', 128, 1000), ('mech_m_M2048_s1', 2048, 1000)]:
        p = model_path(tag)
        if not p:
            print(tag, 'missing'); continue
        pool = lib.sample_tasks(M, torch.Generator().manual_seed(base + M))
        model = load_model(p)
        pm, y, r, g = stats(model, fx, fy, pool)
        base_s = summarize(pm, y, r, g)
        rng = torch.Generator().manual_seed(0)
        oms, rhos = [], []
        for _ in range(2000):
            idx = torch.randint(0, 512, (512,), generator=rng)
            s = summarize(pm[idx], y[idx], r[idx], g[idx])
            oms.append(s['omega']); rhos.append(s['rho'])
        so = torch.tensor(oms).std().item(); sr = torch.tensor(rhos).std().item()
        out[tag] = dict(omega=base_s['omega'], se_omega=so, rho=base_s['rho'], se_rho=sr,
                        rel_se_rho=sr / base_s['rho'],
                        cross_term=(base_s['mse'] - base_s['mse_ridge'] - base_s['Dr']) / base_s['Dr'])
        print(f"{tag:18s} Omega={base_s['omega']:+.3f} se={so:.3f} rho={base_s['rho']:.3f} "
              f"se={sr:.3f} (rel {sr/base_s['rho']:.3f}) identity_residual={out[tag]['cross_term']:+.3f}")
    json.dump(out, open(os.path.join(RES, 'rev_boot.json'), 'w'), indent=1)


def subspace_batch(B, r, gen, scale=1.0):
    """Fresh-task contexts whose 11 demonstrations lie in a random r-dim subspace."""
    w = lib.sample_tasks(B, gen)
    xs = torch.randn(B, lib.K, lib.D, generator=gen)
    if r < lib.D:
        Q, _ = torch.linalg.qr(torch.randn(B, lib.D, lib.D, generator=gen))
        U = Q[:, :, :r]                                        # [B, D, r]
        z = torch.randn(B, lib.K - 1, r, generator=gen)
        xs[:, :-1] = torch.einsum('bkr,bdr->bkd', z, U)
    xs = xs * scale
    ys = (lib.features(xs) * w[:, None, :]).sum(-1) + lib.SIGMA * torch.randn(B, lib.K, generator=gen)
    return xs, ys


def demo():
    setup()
    models = [('mech_m_M2_s1', 2), ('mech_m_M2_s2', 2), ('rr_m_M8_s0', 8), ('m_M12_s0', 12),
              ('mech_m_M16_s1', 16), ('mech_m_M16_s2', 16), ('rr_m_M32_s0', 32),
              ('rr_m_M128_s0', 128), ('mech_m_M2048_s1', 2048), ('mech_m_M2048_s2', 2048)]
    res = []
    for tag, M in models:
        p = model_path(tag)
        if not p:
            print(tag, 'missing'); continue
        model = load_model(p)
        pool = lib.sample_tasks(M, torch.Generator().manual_seed(1000 + M))
        for kind, levels in (('rank', (1, 2, 3, 4)), ('scale', (0.5, 1.0, 1.5, 2.0))):
            for lv in levels:
                gen = torch.Generator().manual_seed(9000 + int(10 * lv))
                if kind == 'rank':
                    xs, ys = subspace_batch(4096, lv, gen)
                else:
                    xs, ys = subspace_batch(4096, lib.D, gen, scale=lv)
                s = summarize(*stats(model, xs, ys, pool))
                s.update(tag=tag, M=M, kind=kind, level=lv)
                res.append(s)
                print(f"{tag:18s} M={M:5d} {kind}={lv}: Omega={s['omega']:+.3f} rho={s['rho']:.2f} "
                      f"dMMSE/ridge={s['dmmse_rho']:.2f} d_r={s['d_r']:.3f} d_g={s['d_g']:.3f} "
                      f"ridgeMSE={s['mse_ridge']:.3f}")
    json.dump(res, open(os.path.join(RES, 'rev_demo.json'), 'w'), indent=1)


def ridge_lam(xs, ys, lam):
    X, Y, xq = xs[:, :-1, :], ys[:, :-1], xs[:, -1, :]
    A = X.transpose(1, 2) @ X + lam * torch.eye(X.shape[-1])[None]
    w = torch.linalg.solve(A, (X.transpose(1, 2) @ Y[..., None])).squeeze(-1)
    return (w * xq).sum(-1)


@torch.no_grad()
def sparse():
    """Sparse family: is the trained network closer to the exact sparse Bayes
    predictor or to ridge regression with the moment-matched Gaussian prior?"""
    setup('sparse')
    lib.SPARSE_S, lib.SPARSE_TAU = 2, 1.0
    gf = torch.Generator().manual_seed(777)
    fx, fy = lib.make_batch(None, 512, gf)
    y = fy[:, -1]
    sb = lib.sbayes_pred(fx, fy)
    lam_mm = lib.SIGMA ** 2 * lib.D / (lib.SPARSE_S * lib.SPARSE_TAU ** 2)
    rmm = ridge_lam(fx, fy, lam_mm)
    r1 = ridge_lam(fx, fy, lib.SIGMA ** 2 / lib.TAU ** 2)
    m = lambda a, b: ((a - b) ** 2).mean().item()
    out = dict(mse_sbayes=m(sb, y), mse_ridge_mm=m(rmm, y), mse_ridge_plain=m(r1, y),
               sep_sb_rmm=m(sb, rmm), lam_mm=lam_mm, models={})
    print(f"sparse refs: MSE sBayes={out['mse_sbayes']:.4f} ridge(mm, lam={lam_mm:.3f})={out['mse_ridge_mm']:.4f} "
          f"ridge(plain)={out['mse_ridge_plain']:.4f} sep(sB,rmm)={out['sep_sb_rmm']:.4f}")
    for tag, M in (('sp_m_M32_s0r', 32), ('sp_m_M128_s0r', 128)):
        pth = model_path(tag)
        if not pth:
            print(tag, 'missing'); continue
        model = load_model(pth)
        pool = lib.sample_tasks(M, torch.Generator().manual_seed(1000 + M))
        f = model(fx, fy)[:, -1]
        g = lib.dmmse_pred(fx, fy, pool)
        rec = dict(M=M, mse=m(f, y), rho_sb=m(f, y) / out['mse_sbayes'], D_sb=m(f, sb), D_rmm=m(f, rmm),
                   D_r1=m(f, r1), D_g=m(f, g), sep=out['sep_sb_rmm'])
        rec['d_sb'] = rec['D_sb'] / out['sep_sb_rmm']; rec['d_rmm'] = rec['D_rmm'] / out['sep_sb_rmm']
        out['models'][tag] = rec
        print(f"{tag}: rho(vs sBayes)={rec['rho_sb']:.3f} D(f,sB)={rec['D_sb']:.4f} D(f,ridge_mm)={rec['D_rmm']:.4f} "
              f"D(f,ridge_plain)={rec['D_r1']:.4f} D(f,dMMSE)={rec['D_g']:.4f} "
              f"normalized: d_sB={rec['d_sb']:.2f} d_ridge={rec['d_rmm']:.2f}")
    json.dump(out, open(os.path.join(RES, 'rev_sparse.json'), 'w'), indent=1)


def demo_summary():
    res = json.load(open(os.path.join(RES, 'rev_demo.json')))
    byM = {}
    for r in res:
        byM.setdefault(r['M'], {}).setdefault((r['kind'], r['level']), []).append(r)
    for M in sorted(byM):
        d = byM[M]
        e1 = sum(r['mse'] for r in d[('rank', 1)]) / len(d[('rank', 1)])
        e4 = sum(r['mse'] for r in d[('rank', 4)]) / len(d[('rank', 4)])
        rr = {lv: [round(r['rho'], 2) for r in d[('rank', lv)]] for lv in (1, 2, 3, 4)}
        om = {lv: [round(r['omega'], 2) for r in d[('rank', lv)]] for lv in (1, 2, 3, 4)}
        sc = {lv: [round(r['rho'], 2) for r in d[('scale', lv)]] for lv in (0.5, 1.5, 2.0)}
        dm = {lv: round(d[('rank', lv)][0]['dmmse_rho'], 2) for lv in (1, 2, 3, 4)}
        print(f'M={M:5d} n={len(d[("rank", 1)])} err r1={e1:.3f} r4={e4:.3f} gain={e1/e4:.1f} rho={rr} dmmse_rho={dm} omega={om} scale_rho={sc}')


@torch.no_grad()
def boot_paired():
    """Bootstrap SE of the difference in Omega and rho between two seeds evaluated on the same contexts,
    against the SE of each estimate alone."""
    setup()
    gf = torch.Generator().manual_seed(777)
    fx, fy = lib.make_batch(None, 512, gf)
    out = {}
    for M, a, b in [(2, 'mech_m_M2_s1', 'mech_m_M2_s2'), (16, 'mech_m_M16_s1', 'mech_m_M16_s2'),
                    (2048, 'mech_m_M2048_s1', 'mech_m_M2048_s2'), (16, 'mech_m_M16_s0r', 'mech_m_M16_s1')]:
        pool = lib.sample_tasks(M, torch.Generator().manual_seed(1000 + M))
        A = stats(load_model(model_path(a)), fx, fy, pool)
        B = stats(load_model(model_path(b)), fx, fy, pool)
        rng = torch.Generator().manual_seed(0)
        d_om, d_rho, oa, ob = [], [], [], []
        for _ in range(2000):
            idx = torch.randint(0, 512, (512,), generator=rng)
            sa = summarize(*(t[idx] for t in A)); sb = summarize(*(t[idx] for t in B))
            d_om.append(sa['omega'] - sb['omega']); d_rho.append(sa['rho'] - sb['rho'])
            oa.append(sa['omega']); ob.append(sb['omega'])
        sa0 = summarize(*A); sb0 = summarize(*B)
        rec = dict(M=M, a=a, b=b, omega_a=sa0['omega'], omega_b=sb0['omega'],
                   se_omega_a=torch.tensor(oa).std().item(), se_omega_b=torch.tensor(ob).std().item(),
                   se_diff_omega=torch.tensor(d_om).std().item(),
                   rho_a=sa0['rho'], rho_b=sb0['rho'], se_diff_rho=torch.tensor(d_rho).std().item())
        out[f'{a}|{b}'] = rec
        print(f"M={M:5d} {a} vs {b}: Omega {rec['omega_a']:+.3f} vs {rec['omega_b']:+.3f}  SE each {rec['se_omega_a']:.3f}/{rec['se_omega_b']:.3f}  "
              f"SE of difference {rec['se_diff_omega']:.3f}; rho {rec['rho_a']:.2f} vs {rec['rho_b']:.2f} SE diff {rec['se_diff_rho']:.3f}")
    json.dump(out, open(os.path.join(RES, 'rev_boot_paired.json'), 'w'), indent=1)


if __name__ == '__main__':
    torch.set_num_threads(1)
    globals()[sys.argv[1]]()
