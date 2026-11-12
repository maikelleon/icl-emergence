"""Mechanistic probes over training checkpoints (paper: mechanism section).

For each checkpoint of an instrumented run:
  * Omega        -- order parameter on the shared fresh eval set
  * wR2          -- R^2 of a linear decoder from the final hidden state to the
                    true task vector w (does the network represent the task?)
  * ymass[l][h]  -- attention mass from the final query token onto y-tokens,
                    per layer and head (information-routing structure)
At the final checkpoint additionally:
  * head ablations -- Omega and fresh-error ratio after zeroing each head.

Output: results/mech_<tag>.jsonl
"""
import argparse, glob, json, math, os, re
import torch
import icl_pilot as lib


def build_eval(M, B=512):
    pool = lib.sample_tasks(M, torch.Generator().manual_seed(1000 + M)) if M > 0 else None
    gf = torch.Generator().manual_seed(777)
    fx, fy = lib.make_batch(None, B, gf)
    fw = None  # regenerate w for decoder probe: re-derive via same generator is
    # not possible post hoc, so draw a separate probe set with known w:
    gp = torch.Generator().manual_seed(4242)
    w = lib.sample_tasks(B, gp)
    xs = torch.randn(B, lib.K, lib.D, generator=gp)
    ys = (xs * w[:, None, :]).sum(-1) + lib.SIGMA * torch.randn(B, lib.K, generator=gp)
    base_r = lib.ridge_pred(fx, fy)
    base_g = lib.dmmse_pred(fx, fy, pool) if pool is not None else base_r
    return (fx, fy, base_r, base_g), (xs, ys, w)


def manual_forward(model, xs, ys, want_attn=False):
    """Replicates ICLTransformer.forward, optionally returning attention maps."""
    B = xs.shape[0]
    tok = torch.empty(B, 2 * lib.K, model.ex.out_features)
    tok[:, 0::2] = model.ex(xs)
    tok[:, 1::2] = model.ey(ys[..., None])
    x = tok + model.pos.weight[None]
    attns = []
    for layer in model.blocks.layers:
        n = layer.norm1(x)
        out, w = layer.self_attn(n, n, n, attn_mask=model.mask,
                                 need_weights=want_attn, average_attn_weights=False)
        if want_attn:
            attns.append(w)          # [B, H, 2K, 2K]
        x = x + out
        n2 = layer.norm2(x)
        x = x + layer.linear2(torch.nn.functional.gelu(layer.linear1(n2)))
    h = model.ln(x)
    preds = model.head(h).squeeze(-1)[:, 0::2]
    return preds, h, attns


def omega_of(preds_last, base_r, base_g):
    Dr = ((preds_last - base_r) ** 2).mean().item()
    Dg = ((preds_last - base_g) ** 2).mean().item()
    return (Dg - Dr) / (Dg + Dr + 1e-12)


def w_decoder_r2(h_final, w):
    """Ridge-decode w from final hidden state; return held-out R^2."""
    n = h_final.shape[0] // 2
    Xtr, Xte = h_final[:n], h_final[n:]
    Ytr, Yte = w[:n], w[n:]
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
    ap.add_argument('--out', default='results')
    args = ap.parse_args()

    lib.D, lib.K = 4, 12
    lib.LAM = (lib.SIGMA / lib.TAU) ** 2
    (fx, fy, base_r, base_g), (px, py, pw) = build_eval(args.M)

    ckpts = sorted(glob.glob(f'{args.out}/ckpt_{args.tag}_*.pt'),
                   key=lambda p: int(re.search(r'_(\d+)\.pt$', p).group(1)))
    if not ckpts:
        raise SystemExit(f'no checkpoints for {args.tag}')
    outpath = f'{args.out}/mech_{args.tag}.jsonl'
    f = open(outpath, 'w')

    model = lib.ICLTransformer(args.layers, args.dmodel, args.heads)
    model.eval()

    # parity self-check on the first checkpoint
    model.load_state_dict(torch.load(ckpts[0], weights_only=True))
    p_manual, _, _ = manual_forward(model, fx[:32], fy[:32])
    p_ref = model(fx[:32], fy[:32])
    assert torch.allclose(p_manual, p_ref, atol=1e-4), 'manual forward mismatch'

    for cp in ckpts:
        step = int(re.search(r'_(\d+)\.pt$', cp).group(1))
        model.load_state_dict(torch.load(cp, weights_only=True))
        preds, _, attns = manual_forward(model, fx, fy, want_attn=True)
        om = omega_of(preds[:, -1], base_r, base_g)
        _, h_probe, _ = manual_forward(model, px, py)
        r2 = w_decoder_r2(h_probe[:, -2, :], pw)   # final x-token position
        qrow = 2 * lib.K - 2
        ymass = [a[:, :, qrow, 1::2].sum(-1).mean(0).tolist() for a in attns]
        rec = dict(type='probe', step=step, omega=round(om, 4),
                   wR2=round(r2, 4), ymass=[[round(v, 4) for v in l] for l in ymass])
        f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec), flush=True)

    # head ablations at the final checkpoint
    model.load_state_dict(torch.load(ckpts[-1], weights_only=True))
    preds, _, _ = manual_forward(model, fx, fy)
    base_mse = ((preds[:, -1] - fy[:, -1]) ** 2).mean().item()
    dh = args.dmodel // args.heads
    for li, layer in enumerate(model.blocks.layers):
        for hi in range(args.heads):
            keep = layer.self_attn.out_proj.weight.data.clone()
            layer.self_attn.out_proj.weight.data[:, hi * dh:(hi + 1) * dh] = 0
            p2, _, _ = manual_forward(model, fx, fy)
            om2 = omega_of(p2[:, -1], base_r, base_g)
            mse2 = ((p2[:, -1] - fy[:, -1]) ** 2).mean().item()
            layer.self_attn.out_proj.weight.data = keep
            rec = dict(type='ablate', layer=li, head=hi,
                       omega=round(om2, 4), mse_ratio_vs_intact=round(mse2 / base_mse, 3))
            f.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)
    f.close()
    print('wrote', outpath)


if __name__ == '__main__':
    main()
