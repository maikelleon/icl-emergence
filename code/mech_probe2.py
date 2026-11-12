"""Mechanistic probes over training checkpoints (revision version).

Extends mech_probe.py with (i) arbitrary architecture, seed, and pool draw,
(ii) mean-ablation alongside zero-ablation, and (iii) per-ablation error ratios
relative to ridge as well as to the intact network. The original probe logic
(order parameter, task-vector decoder, attention mass on y tokens from the final
query position) is unchanged, so outputs are comparable with the first campaign.

For each checkpoint of an instrumented run:
  * omega        -- order parameter on the shared fresh evaluation set
  * wR2          -- held-out R^2 of a linear decoder from the final x-token
                    hidden state to the task vector w (separate probe set)
  * ymass[l][h]  -- attention mass from the final query token onto y tokens
At the final checkpoint, for every head and for both ablation modes:
  * zero: the head's output (before the output projection) is set to zero
  * mean: the head's output is replaced by its mean over the evaluation batch,
          position by position (removes context-specific information only)

Output: results/mech2_<tag>.jsonl
"""
import argparse, glob, json, math, re
import torch
import icl_pilot as lib


def build_eval(M, pool_seed_base, B=512):
    pool = (lib.sample_tasks(M, torch.Generator().manual_seed(pool_seed_base + M))
            if M > 0 else None)
    gf = torch.Generator().manual_seed(777)
    fx, fy = lib.make_batch(None, B, gf)
    gp = torch.Generator().manual_seed(4242)
    w = lib.sample_tasks(B, gp)
    xs = torch.randn(B, lib.K, lib.D, generator=gp)
    ys = (lib.features(xs) * w[:, None, :]).sum(-1) + lib.SIGMA * torch.randn(B, lib.K, generator=gp)
    base_r = lib.ridge_pred(fx, fy)
    base_g = lib.dmmse_pred(fx, fy, pool) if pool is not None else base_r
    return (fx, fy, base_r, base_g), (xs, ys, w)


def forward(model, xs, ys, ablate=None, want_attn=False):
    """Manual replica of ICLTransformer.forward with explicit per-head attention.
    ablate: None or (layer, head, mode) with mode in {'zero', 'mean'}."""
    B = xs.shape[0]
    T = 2 * lib.K
    dm = model.ex.out_features
    tok = torch.empty(B, T, dm)
    tok[:, 0::2] = model.ex(xs)
    tok[:, 1::2] = model.ey(ys[..., None])
    x = tok + model.pos.weight[None]
    attns = []
    for li, layer in enumerate(model.blocks.layers):
        sa = layer.self_attn
        H = sa.num_heads
        dh = dm // H
        n = layer.norm1(x)
        qkv = n @ sa.in_proj_weight.T + sa.in_proj_bias
        q, k, v = qkv.split(dm, dim=-1)
        q = q.view(B, T, H, dh).transpose(1, 2)
        k = k.view(B, T, H, dh).transpose(1, 2)
        v = v.view(B, T, H, dh).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dh) + model.mask
        a = torch.softmax(scores, dim=-1)                      # [B, H, T, T]
        o = a @ v                                              # [B, H, T, dh]
        if ablate is not None and ablate[0] == li:
            h = ablate[1]
            if ablate[2] == 'zero':
                o[:, h] = 0.0
            else:
                o[:, h] = o[:, h].mean(0, keepdim=True)
        if want_attn:
            attns.append(a)
        o = o.transpose(1, 2).reshape(B, T, dm)
        x = x + sa.out_proj(o)
        n2 = layer.norm2(x)
        x = x + layer.linear2(torch.nn.functional.gelu(layer.linear1(n2)))
    h = model.ln(x)
    preds = model.head(h).squeeze(-1)[:, 0::2]
    return preds, h, attns


def omega_of(p, r, g):
    Dr = ((p - r) ** 2).mean().item()
    Dg = ((p - g) ** 2).mean().item()
    return (Dg - Dr) / (Dg + Dr + 1e-12)


def w_decoder_r2(h, w):
    n = h.shape[0] // 2
    Xtr, Xte, Ytr, Yte = h[:n], h[n:], w[:n], w[n:]
    A = Xtr.T @ Xtr + 1e-2 * torch.eye(Xtr.shape[1])
    W = torch.linalg.solve(A, Xtr.T @ Ytr)
    resid = ((Xte @ W - Yte) ** 2).sum()
    tot = ((Yte - Ytr.mean(0)) ** 2).sum()
    return (1 - resid / tot).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--M', type=int, required=True)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--dmodel', type=int, default=64)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--pool-seed-base', type=int, default=1000)
    ap.add_argument('--every', type=int, default=500, help='probe checkpoints at this cadence')
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    lib.D, lib.K = 4, 12
    lib.LAM = (lib.SIGMA / lib.TAU) ** 2
    (fx, fy, base_r, base_g), (px, py, pw) = build_eval(args.M, args.pool_seed_base)

    ckpts = sorted(glob.glob(f'{args.out}/ckpt_{args.tag}_*.pt'),
                   key=lambda p: int(re.search(r'_(\d+)\.pt$', p).group(1)))
    ckpts = [c for c in ckpts if int(re.search(r'_(\d+)\.pt$', c).group(1)) % args.every == 0]
    if not ckpts:
        raise SystemExit(f'no checkpoints for {args.tag}')
    f = open(f'{args.out}/mech2_{args.tag}.jsonl', 'w')
    model = lib.ICLTransformer(args.layers, args.dmodel, args.heads)
    model.eval()

    model.load_state_dict(torch.load(ckpts[-1], weights_only=True))
    p_manual, _, _ = forward(model, fx[:64], fy[:64])
    assert torch.allclose(p_manual, model(fx[:64], fy[:64]), atol=1e-4), 'forward mismatch'

    qrow = 2 * lib.K - 2
    for cp in ckpts:
        step = int(re.search(r'_(\d+)\.pt$', cp).group(1))
        model.load_state_dict(torch.load(cp, weights_only=True))
        preds, _, attns = forward(model, fx, fy, want_attn=True)
        om = omega_of(preds[:, -1], base_r, base_g)
        _, hp, _ = forward(model, px, py)
        r2 = w_decoder_r2(hp[:, -2, :], pw)
        ymass = [a[:, :, qrow, 1::2].sum(-1).mean(0).tolist() for a in attns]
        f.write(json.dumps(dict(type='probe', step=step, omega=round(om, 4), wR2=round(r2, 4),
                                ymass=[[round(v, 4) for v in l] for l in ymass])) + '\n')

    model.load_state_dict(torch.load(ckpts[-1], weights_only=True))
    preds, _, _ = forward(model, fx, fy)
    y = fy[:, -1]
    base_mse = ((preds[:, -1] - y) ** 2).mean().item()
    ridge_mse = ((base_r - y) ** 2).mean().item()
    # pool-context set (the training distribution), as in icl_pilot.py: seed 778
    qx = qy = None
    if args.M > 0:
        pool = lib.sample_tasks(args.M, torch.Generator().manual_seed(args.pool_seed_base + args.M))
        qx, qy = lib.make_batch(pool, 512, torch.Generator().manual_seed(778))
        qd = lib.dmmse_pred(qx, qy, pool)
        qp, _, _ = forward(model, qx, qy)
        pool_mse = ((qp[:, -1] - qy[:, -1]) ** 2).mean().item()
        pool_dmmse = ((qd - qy[:, -1]) ** 2).mean().item()
    f.write(json.dumps(dict(type='intact', mse=base_mse, rho=base_mse / ridge_mse,
                            omega=omega_of(preds[:, -1], base_r, base_g),
                            mse_pool=pool_mse if qx is not None else None,
                            mse_pool_dmmse=pool_dmmse if qx is not None else None)) + '\n')
    for li in range(args.layers):
        for hi in range(args.heads):
            for mode in ('zero', 'mean'):
                p2, _, _ = forward(model, fx, fy, ablate=(li, hi, mode))
                mse2 = ((p2[:, -1] - y) ** 2).mean().item()
                rec = dict(type='ablate', layer=li, head=hi, mode=mode,
                           omega=round(omega_of(p2[:, -1], base_r, base_g), 4),
                           mse_ratio_vs_intact=round(mse2 / base_mse, 4),
                           rho=round(mse2 / ridge_mse, 4))
                if qx is not None:
                    q2, _, _ = forward(model, qx, qy, ablate=(li, hi, mode))
                    pm2 = ((q2[:, -1] - qy[:, -1]) ** 2).mean().item()
                    rec['pool_mse_ratio_vs_intact'] = round(pm2 / pool_mse, 4)
                    rec['pool_mse'] = round(pm2, 5)
                f.write(json.dumps(rec) + '\n')
    f.close()
    print('wrote', f'{args.out}/mech2_{args.tag}.jsonl')


if __name__ == '__main__':
    main()
