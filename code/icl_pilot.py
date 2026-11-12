"""
Pilot: phase structure of in-context learning in small transformers.

Setup (Raventos et al.-style): in-context linear regression with a FINITE
pretraining task pool of size M (task diversity). A memorizing solution
matches dMMSE (Bayes under the discrete pool prior); a true ICL solution
matches ridge regression (Bayes under the continuous Gaussian prior).
Order parameter in [-1, +1]: -1 = model predictions align with dMMSE
(memorization phase), +1 = align with ridge (ICL phase).
"""
import argparse, json, math, os, time
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("ICL_THREADS", "2")))

D = 4           # input dimension (set from CLI in main)
K = 12          # (x, y) pairs per context (set from CLI in main)
SIGMA = 0.25    # observation noise std
TAU = 1.0       # task prior std  (w ~ N(0, TAU^2 I))
LAM = (SIGMA / TAU) ** 2  # ridge regularizer = Bayes posterior mean


# ---------------------------------------------------------------- data
FAMILY = 'linear'   # set from CLI: 'linear', 'sparse', or 'nonlinear'
SPARSE_S = 2        # nonzero coordinates for the sparse family
SPARSE_TAU = 1.0    # active-weight prior std for the sparse family
                    # (sqrt(2) matches the dense family's signal power s*tau'^2 = d*tau^2)
# Nonlinear family (random-feature regression): f(x) = a^T phi(x), with a fixed
# random Fourier feature map phi(x) = sqrt(2/P) cos(Wx + b) shared by all tasks and
# a ~ N(0, NL_TAU^2 I_P). The exact Bayes predictor is ridge regression in feature
# space (equivalently, kernel ridge regression), so both poles stay exact.
NL_P = 4            # number of random features (task dimension of the family)
NL_SCALE = 1.0      # input lengthscale of the features (W ~ N(0, I / NL_SCALE^2))
NL_TAU = 2.0        # feature-weight prior std; 2.0 gives E f(x)^2 = 4 = d * tau^2
NL_W, NL_B = None, None


def init_features(seed=4321):
    """Draw the fixed feature map once; it is part of the family definition."""
    global NL_W, NL_B
    g = torch.Generator().manual_seed(seed)
    NL_W = torch.randn(NL_P, D, generator=g) / NL_SCALE
    NL_B = 2 * math.pi * torch.rand(NL_P, generator=g)


def features(xs):
    """Regression features: identity for the linear families, phi(x) otherwise."""
    if FAMILY != 'nonlinear':
        return xs
    return math.sqrt(2.0 / NL_P) * torch.cos(xs @ NL_W.T + NL_B)


def prior_lam():
    """Ridge regularizer sigma^2 / tau^2 of the family's continuous prior."""
    return (SIGMA / NL_TAU) ** 2 if FAMILY == 'nonlinear' else LAM


def sample_tasks(n, gen):
    if FAMILY == 'nonlinear':
        return torch.randn(n, NL_P, generator=gen) * NL_TAU
    w = torch.randn(n, D, generator=gen) * TAU
    if FAMILY == 'sparse':
        w = w * (SPARSE_TAU / TAU)   # pure rescale: RNG stream identical to tau'=1
        mask = torch.zeros(n, D)
        for i in range(n):
            idx = torch.randperm(D, generator=gen)[:SPARSE_S]
            mask[i, idx] = 1.0
        w = w * mask
    return w


def make_batch(W, B, gen, d_active=None):
    """W: [M, D] pool (or None => fresh tasks each time).
    d_active: curriculum — zero out input dims beyond d_active."""
    if W is None:
        w = sample_tasks(B, gen)
    else:
        idx = torch.randint(0, W.shape[0], (B,), generator=gen)
        w = W[idx]
    xs = torch.randn(B, K, D, generator=gen)
    if d_active is not None and d_active < D:
        xs[..., d_active:] = 0.0
    ys = (features(xs) * w[:, None, :]).sum(-1) + SIGMA * torch.randn(B, K, generator=gen)
    return xs, ys


# ------------------------------------------------------------ baselines
def ridge_pred(xs, ys):
    """Bayes-optimal under continuous Gaussian prior; last-position pred.
    For the nonlinear family this is ridge regression in feature space."""
    F = features(xs)
    X, Y, xq = F[:, :-1, :], ys[:, :-1], F[:, -1, :]
    P = X.shape[-1]
    A = X.transpose(1, 2) @ X + prior_lam() * torch.eye(P)[None]
    b = (X.transpose(1, 2) @ Y[..., None]).squeeze(-1)
    w = torch.linalg.solve(A, b)
    return (w * xq).sum(-1)


def dmmse_pred(xs, ys, W):
    """Bayes-optimal under the discrete pool prior; last-position pred."""
    F = features(xs)
    X, Y, xq = F[:, :-1, :], ys[:, :-1], F[:, -1, :]
    f = torch.einsum('bkd,md->bkm', X, W)                # [B, K-1, M]
    ll = -0.5 / SIGMA**2 * ((Y[..., None] - f) ** 2).sum(1)  # [B, M]
    p = torch.softmax(ll, dim=-1)
    q = torch.einsum('bd,md->bm', xq, W)                 # [B, M]
    return (p * q).sum(-1)


def sbayes_pred(xs, ys):
    """Exact Bayes under the sparse prior: enumerate supports, mix Gaussian
    posteriors by marginal likelihood. ICL pole for the sparse family."""
    import itertools, math as _m
    X, Y, xq = xs[:, :-1, :], ys[:, :-1], xs[:, -1, :]
    B, n = Y.shape
    preds, logmls = [], []
    for S in itertools.combinations(range(D), SPARSE_S):
        Si = torch.tensor(S)
        Xs = X[:, :, Si]                                   # [B, n, s]
        C = SPARSE_TAU**2 * (Xs @ Xs.transpose(1, 2)) + SIGMA**2 * torch.eye(n)[None]
        L = torch.linalg.cholesky(C)
        a = torch.cholesky_solve(Y[..., None], L).squeeze(-1)   # C^{-1} y
        logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2)).sum(-1)
        logml = -0.5 * ((Y * a).sum(-1) + logdet + n * _m.log(2 * _m.pi))
        A = Xs.transpose(1, 2) @ Xs + (SIGMA**2 / SPARSE_TAU**2) * torch.eye(SPARSE_S)[None]
        w = torch.linalg.solve(A, (Xs.transpose(1, 2) @ Y[..., None])).squeeze(-1)
        preds.append((w * xq[:, Si]).sum(-1))
        logmls.append(logml)
    P = torch.softmax(torch.stack(logmls, -1), dim=-1)     # [B, n_supports]
    F = torch.stack(preds, -1)
    return (P * F).sum(-1)


# ---------------------------------------------------------------- model
class ICLTransformer(nn.Module):
    def __init__(self, layers, dm, heads):
        super().__init__()
        self.ex = nn.Linear(D, dm)
        self.ey = nn.Linear(1, dm)
        self.pos = nn.Embedding(2 * K, dm)
        enc = nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=0.0,
                                         activation='gelu', batch_first=True,
                                         norm_first=True)
        self.blocks = nn.TransformerEncoder(enc, layers)
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, 1)
        mask = torch.triu(torch.full((2 * K, 2 * K), float('-inf')), diagonal=1)
        self.register_buffer('mask', mask)

    def forward(self, xs, ys):
        B = xs.shape[0]
        tok = torch.empty(B, 2 * K, self.ex.out_features)
        tok[:, 0::2] = self.ex(xs)
        tok[:, 1::2] = self.ey(ys[..., None])
        tok = tok + self.pos.weight[None]
        h = self.blocks(tok, mask=self.mask)
        return self.head(self.ln(h[:, 0::2])).squeeze(-1)   # [B, K] preds


# ----------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, ev):
    model.eval()
    out = {}
    for name, (xs, ys, base_r, base_d) in ev.items():
        pm = model(xs, ys)[:, -1]
        y = ys[:, -1]
        out[f'mse_{name}_model'] = ((pm - y) ** 2).mean().item()
        out[f'mse_{name}_ridge'] = ((base_r - y) ** 2).mean().item()
        out[f'mse_{name}_dmmse'] = ((base_d - y) ** 2).mean().item()
        Dr = ((pm - base_r) ** 2).mean().item()
        Dd = ((pm - base_d) ** 2).mean().item()
        out[f'order_{name}'] = (Dd - Dr) / (Dd + Dr + 1e-12)
        # prediction-level distances (added in revision): squared distance of the
        # model to each reference and between the two references themselves
        out[f'Dr_{name}'] = Dr
        out[f'Dg_{name}'] = Dd
        out[f'Drg_{name}'] = ((base_r - base_d) ** 2).mean().item()
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--M', type=int, required=True, help='task pool size; 0 = infinite')
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dmodel', type=int, default=64)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warmup', type=int, default=300)
    ap.add_argument('--eval-every', type=int, default=1000)
    ap.add_argument('--eval-B', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='run')
    ap.add_argument('--out', type=str, default='results')
    ap.add_argument('--dim', type=int, default=4)
    ap.add_argument('--K', type=int, default=12)
    ap.add_argument('--curr', action='store_true',
                    help='dimension curriculum: 2 -> D over first 40%% of steps')
    ap.add_argument('--curr-ramp', type=int, default=0,
                    help='absolute curriculum ramp in steps (0 = 40%% of --steps)')
    ap.add_argument('--schedule', choices=['cosine', 'const'], default='cosine',
                    help='LR after warmup: cosine decay to 10%% or constant')
    ap.add_argument('--family', choices=['linear', 'sparse', 'nonlinear'], default='linear')
    ap.add_argument('--nl-p', type=int, default=4, help='random features (nonlinear family)')
    ap.add_argument('--nl-scale', type=float, default=1.0, help='feature lengthscale')
    ap.add_argument('--nl-tau', type=float, default=2.0, help='feature-weight prior std')
    ap.add_argument('--save-final', action='store_true', help='save the final model')
    ap.add_argument('--sparse-s', type=int, default=2)
    ap.add_argument('--sparse-tau', type=float, default=1.0,
                    help='active-weight prior std for the sparse family; '
                         '1.4142135623730951 matches the dense signal power')
    ap.add_argument('--pool-seed-base', type=int, default=1000,
                    help='base seed for the frozen task pool (pool seed = base + M); '
                         '1000 reproduces the main campaign, other values draw a fresh pool')
    ap.add_argument('--save-ckpt-every', type=int, default=0,
                    help='save model checkpoints every N steps (0 = never)')
    ap.add_argument('--auto-resume', action='store_true',
                    help='save a resume bundle at every eval and continue from it on restart')
    ap.add_argument('--baselines-check', action='store_true')
    args = ap.parse_args()

    global D, K, LAM, FAMILY, SPARSE_S, SPARSE_TAU, NL_P, NL_SCALE, NL_TAU
    D, K = args.dim, args.K
    LAM = (SIGMA / TAU) ** 2
    FAMILY, SPARSE_S, SPARSE_TAU = args.family, args.sparse_s, args.sparse_tau
    NL_P, NL_SCALE, NL_TAU = args.nl_p, args.nl_scale, args.nl_tau
    if FAMILY == 'nonlinear':
        init_features()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    # task pool: deterministic per M so every model size sees the same pool
    pool = None
    if args.M > 0:
        pool = sample_tasks(args.M, torch.Generator().manual_seed(args.pool_seed_base + args.M))

    # fixed eval sets, identical across all runs of a family
    bayes = sbayes_pred if FAMILY == 'sparse' else ridge_pred
    gf = torch.Generator().manual_seed(777)
    fx, fy = make_batch(None, args.eval_B, gf)                      # fresh tasks
    ev = {'fresh': (fx, fy, bayes(fx, fy),
                    dmmse_pred(fx, fy, pool) if pool is not None else bayes(fx, fy))}
    if pool is not None:
        gp = torch.Generator().manual_seed(778)
        px, py = make_batch(pool, args.eval_B, gp)                  # pool tasks
        ev['pool'] = (px, py, bayes(px, py), dmmse_pred(px, py, pool))

    if args.baselines_check:
        for name, (xs, ys, br, bd) in ev.items():
            y = ys[:, -1]
            print(f'[{name}] mse ridge={((br-y)**2).mean():.4f} '
                  f'dmmse={((bd-y)**2).mean():.4f} zero={((0*y-y)**2).mean():.4f} '
                  f'noise-floor={SIGMA**2:.4f}')
        return

    model = ICLTransformer(args.layers, args.dmodel, args.heads)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_fn(s):  # linear warmup, then cosine to 10% or constant
        wu = min(1.0, (s + 1) / args.warmup)
        if args.schedule == 'const':
            return wu
        cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / args.steps))
        return wu * cos
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    def d_active_at(s):
        if not args.curr:
            return None
        ramp = args.curr_ramp if args.curr_ramp > 0 else 0.4 * args.steps
        return min(D, 2 + int((s / ramp) * (D - 2 + 1)))

    path = os.path.join(args.out, f'{args.tag}.jsonl')
    rpath = os.path.join(args.out, f'resume_{args.tag}.pt')
    start_step = 1
    if args.auto_resume and os.path.exists(rpath):
        bundle = torch.load(rpath, weights_only=False)
        if bundle['step'] < args.steps:
            model.load_state_dict(bundle['model'])
            opt.load_state_dict(bundle['opt'])
            sched.load_state_dict(bundle['sched'])
            start_step = bundle['step'] + 1
            print(f'resumed {args.tag} from step {bundle["step"]}')

    meta = dict(type='meta', M=args.M, layers=args.layers, dmodel=args.dmodel,
                heads=args.heads, steps=args.steps, batch=args.batch,
                lr=args.lr, seed=args.seed, params=nparams, D=D, K=K,
                sigma=SIGMA, curr=args.curr, family=FAMILY, tag=args.tag,
                pool_seed_base=args.pool_seed_base, sparse_tau=SPARSE_TAU,
                nl_p=NL_P, nl_scale=NL_SCALE, nl_tau=NL_TAU,
                threads=torch.get_num_threads(), torch_version=torch.__version__)
    with open(path, 'a' if start_step > 1 else 'w') as f:
        f.write(json.dumps(meta) + '\n')
    print(json.dumps(meta))

    g = torch.Generator().manual_seed(args.seed + 1 + 31 * (start_step - 1))
    t0, loss_acc = time.time(), 0.0
    for step in range(start_step, args.steps + 1):
        xs, ys = make_batch(pool, args.batch, g, d_active=d_active_at(step))
        preds = model(xs, ys)
        loss = ((preds - ys) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        loss_acc += loss.item()
        if args.save_ckpt_every and (step % args.save_ckpt_every == 0 or step == args.steps):
            torch.save(model.state_dict(),
                       os.path.join(args.out, f'ckpt_{args.tag}_{step}.pt'))
        if step % args.eval_every == 0 or step == args.steps:
            rec = dict(type='eval', step=step,
                       train_loss=loss_acc / args.eval_every,
                       secs=round(time.time() - t0, 1))
            rec.update(evaluate(model, ev))
            loss_acc = 0.0
            with open(path, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)
            if args.auto_resume:
                torch.save(dict(step=step, model=model.state_dict(),
                                opt=opt.state_dict(), sched=sched.state_dict()),
                           rpath + '.tmp')
                os.replace(rpath + '.tmp', rpath)
    if args.save_final:
        torch.save(model.state_dict(), os.path.join(args.out, f'final_{args.tag}.pt'))
    print(f'done {args.tag} in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
