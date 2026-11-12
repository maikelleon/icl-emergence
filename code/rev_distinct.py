"""Post hoc distinguishability of pool tasks: dMMSE posterior over pool tasks on
pool contexts, per family, at M = 8, 16, 32 (original pool draw)."""
import json, math, torch, icl_pilot as lib
def setup(family, tau_s=1.0):
    lib.D, lib.K = 4, 12
    lib.FAMILY = family; lib.SPARSE_S = 2; lib.SPARSE_TAU = tau_s
    lib.NL_P, lib.NL_SCALE, lib.NL_TAU = 4, 1.0, 2.0
    if family == 'nonlinear': lib.init_features()
out = {}
for name, fam, ts in (('dense', 'linear', 1.0), ('sparse', 'sparse', 1.0),
                      ('matched', 'sparse', math.sqrt(2)), ('nonlinear', 'nonlinear', 1.0)):
    setup(fam, ts)
    for M in (8, 16, 32):
        pool = lib.sample_tasks(M, torch.Generator().manual_seed(1000 + M))
        g = torch.Generator().manual_seed(778)
        B = 4096
        idx = torch.randint(0, M, (B,), generator=g)          # replicate make_batch's draws
        w = pool[idx]
        xs = torch.randn(B, lib.K, lib.D, generator=g)
        ys = (lib.features(xs) * w[:, None, :]).sum(-1) + lib.SIGMA * torch.randn(B, lib.K, generator=g)
        F = lib.features(xs)
        Hs = []
        for k in range(1, 12):
            f = torch.einsum('bkd,md->bkm', F[:, :k], pool)
            ll = -0.5 / lib.SIGMA ** 2 * ((ys[:, :k, None] - f) ** 2).sum(1)
            p = torch.softmax(ll, -1)
            Hs.append((-(p * torch.log(p + 1e-30)).sum(-1).mean() / math.log(M)).item())
            if k == 11:
                pc = p[torch.arange(B), idx].mean().item()
        out[f'{name}_M{M}'] = dict(meanH=sum(Hs) / len(Hs), H1=Hs[0], H11=Hs[-1], p_correct_11=pc)
        print(f'{name:9s} M={M:3d} meanH={sum(Hs)/len(Hs):.3f} H(k=1)={Hs[0]:.2f} p_correct(k=11)={pc:.4f}')
json.dump(out, open('results/rev_distinct.json', 'w'), indent=1)
