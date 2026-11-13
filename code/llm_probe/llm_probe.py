"""In-context regression across the pretraining checkpoints of open language models.

Companion probe to "When Does In-Context Learning Emerge, and Does It Endure? Evidence from
Small Transformers". For each model size and pretraining checkpoint of the Pythia suite, the
probe shows the model k noisy (x, y) pairs from a random linear function, written as text, and
asks for y at a new x. Each prediction is compared with exact references computed from the
same rounded numbers that appear in the prompt:

  bayes    posterior mean under the task prior (the Bayes-optimal predictor)
  ols      ordinary least squares on the demonstrations
  ctxmean  mean of the demonstrated outputs (uses no information about x)
  nn1      output of the demonstration whose input is closest to the query

Two summary statistics follow the paper: the error ratio rho = MSE(model) / MSE(bayes), and an
in-context score S = (MSE(ctxmean) - MSE(model)) / (MSE(ctxmean) - MSE(bayes)), which is 0 for
a model that only reproduces the average output and 1 for the Bayes-optimal predictor. Because
the same tasks are shown to every checkpoint, the change of S between checkpoints is a paired
comparison, which is how the probe tests whether in-context regression, once formed, persists.

Usage
  python llm_probe.py selftest                                  # no downloads, about a minute
  python llm_probe.py run --models 70m 160m                     # default checkpoints, k = 10
  python llm_probe.py run --models 410m --steps 1000 143000     # a subset of checkpoints
  python llm_probe.py summarize                                 # table, CSV, and figure

Results are written to results/ (one JSONL file per model and checkpoint); a run that stops can
be restarted with the same command and skips what is already done. Each checkpoint is deleted
after its evaluation unless --keep is given, so the disk needs only about 1 GB free.
"""
import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np

SIZES = {'70m': 'EleutherAI/pythia-70m-deduped', '160m': 'EleutherAI/pythia-160m-deduped',
         '410m': 'EleutherAI/pythia-410m-deduped', '1b': 'EleutherAI/pythia-1b-deduped',
         '1.4b': 'EleutherAI/pythia-1.4b-deduped', '2.8b': 'EleutherAI/pythia-2.8b-deduped'}
DEFAULT_STEPS = [512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 143000]

# Task prior: y = a + w x + noise, x ~ U[0, 10]; (a, w) ~ N(MU0, diag(SD0^2)); noise sd SIGMA.
MU0 = np.array([30.0, 4.0])
SD0 = np.array([8.0, 1.5])
SIGMA = 1.0
X_LO, X_HI = 0.0, 10.0


# ------------------------------------------------------------------------------------------ tasks
def make_tasks(n_tasks, k, seed):
    """Tasks with k demonstrations and one query; numbers rounded to two decimals, as shown."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(n_tasks):
        a, w = rng.normal(MU0, SD0)
        x = np.round(rng.uniform(X_LO, X_HI, k + 1), 2)
        y = np.round(a + w * x + rng.normal(0.0, SIGMA, k + 1), 2)
        tasks.append({'id': i, 'k': k, 'a': float(a), 'w': float(w),
                      'x': x[:k].tolist(), 'y': y[:k].tolist(),
                      'xq': float(x[k]), 'yq': float(y[k])})
    return tasks


def references(t):
    x, y, xq = np.array(t['x']), np.array(t['y']), t['xq']
    phi = np.stack([np.ones_like(x), x], 1)
    pq = np.array([1.0, xq])
    prec0 = np.diag(1.0 / SD0 ** 2)
    cov_n = np.linalg.inv(prec0 + phi.T @ phi / SIGMA ** 2)
    mu_n = cov_n @ (prec0 @ MU0 + phi.T @ y / SIGMA ** 2)
    beta_ols = np.linalg.lstsq(phi, y, rcond=None)[0]
    return {'bayes': float(pq @ mu_n), 'ols': float(pq @ beta_ols), 'ctxmean': float(y.mean()),
            'nn1': float(y[np.argmin(np.abs(x - xq))])}


def prompt(t):
    lines = [f'Input: {xi:.2f}\nOutput: {yi:.2f}\n' for xi, yi in zip(t['x'], t['y'])]
    return '\n'.join(lines) + f'\nInput: {t["xq"]:.2f}\nOutput:'


NUM = re.compile(r'-?\d+(?:\.\d+)?')


def parse(text, t):
    """First number before the first line break; None if there is none."""
    m = NUM.search(text.split('\n')[0])
    return float(m.group(0)) if m else None


# ------------------------------------------------------------------------------------------ model
def load_model(repo, step, cache_dir, device):
    import torch
    from transformers import AutoModelForCausalLM
    kw = dict(revision=f'step{step}', cache_dir=cache_dir)
    if device == 'cuda':
        kw['device_map'] = 'cuda'       # load straight onto the GPU (needs accelerate)
    dtype = torch.float16 if device == 'cuda' else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=dtype, **kw)
    return model.to(device).eval()


def predict(model, tok, prompts, device, batch, max_new_tokens=8):
    import torch
    out = []
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        new = gen[:, enc['input_ids'].shape[1]:]
        out.extend(tok.batch_decode(new, skip_special_tokens=True))
    return out


def evaluate(model, tok, tasks, device, batch):
    texts = predict(model, tok, [prompt(t) for t in tasks], device, batch)
    rows = []
    for t, text in zip(tasks, texts):
        ref = references(t)
        lo, hi = min(t['y']), max(t['y'])
        span = hi - lo
        p = parse(text, t)
        parsed = p is not None
        if not parsed:
            p = ref['ctxmean']            # conservative: an unparseable answer earns no credit
        clipped = not (lo - span <= p <= hi + span)
        p = min(max(p, lo - span), hi + span)
        rows.append({'id': t['id'], 'k': t['k'], 'yq': t['yq'], 'pred': p, 'raw': text[:40],
                     'parsed': parsed, 'clipped': clipped, **{f'ref_{n}': v for n, v in ref.items()}})
    return rows


# ------------------------------------------------------------------------------------------ run
def cmd_run(args):
    import torch
    from transformers import AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    if device == 'cpu':
        torch.set_num_threads(args.threads or os.cpu_count())
    os.makedirs(args.out, exist_ok=True)
    tasks = {k: make_tasks(args.n_tasks, k, args.seed + k) for k in args.k}
    plan = [(m, s) for m in args.models for s in args.steps]
    print(f'device={device}; {len(plan)} checkpoints x {len(args.k)} context lengths x {args.n_tasks} tasks')
    t_start, done = time.time(), 0
    for size, step in plan:
        path = os.path.join(args.out, f'probe_{size}_step{step}.jsonl')
        if os.path.exists(path):
            print(f'skip {size} step {step} (done)'); done += 1; continue
        repo = SIZES[size]
        cache = os.path.join(args.cache, f'{size}_step{step}')
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(repo, cache_dir=os.path.join(args.cache, 'tokenizers'))
        tok.padding_side = 'left'
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = load_model(repo, step, cache, device)
        rows = []
        for k in args.k:
            for r in evaluate(model, tok, tasks[k], device, args.batch):
                rows.append({'model': size, 'step': step, **r})
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
        os.replace(tmp, path)
        del model; gc.collect()
        if not args.keep:
            shutil.rmtree(cache, ignore_errors=True)
        done += 1
        s = summarize_rows(rows)
        eta = (time.time() - t_start) / max(done, 1) * (len(plan) - done)
        print(f'{size:5s} step {step:6d}: S={s["S"]:+.3f} rho={s["rho"]:.2f} parsed={s["parse_rate"]:.2f} '
              f'({time.time() - t0:.0f}s; about {eta / 60:.0f} min left)', flush=True)


# ------------------------------------------------------------------------------------------ summarize
def summarize_rows(rows, boot=0, seed=0):
    y = np.array([r['yq'] for r in rows]); f = np.array([r['pred'] for r in rows])
    R = {n: np.array([r[f'ref_{n}'] for r in rows]) for n in ('bayes', 'ols', 'ctxmean', 'nn1')}

    def stats(idx):
        mse = lambda p: float(np.mean((p[idx] - y[idx]) ** 2))
        m, b, c = mse(f), mse(R['bayes']), mse(R['ctxmean'])
        return m, b, c, (c - m) / (c - b), m / b
    m, b, c, S, rho = stats(np.arange(len(y)))
    out = {'n': len(y), 'mse_model': m, 'mse_bayes': b, 'mse_ctxmean': c, 'S': S, 'rho': rho,
           'mse_ols': float(np.mean((R['ols'] - y) ** 2)), 'mse_nn1': float(np.mean((R['nn1'] - y) ** 2)),
           'parse_rate': float(np.mean([r['parsed'] for r in rows])),
           'clip_rate': float(np.mean([r['clipped'] for r in rows]))}
    for n in ('bayes', 'ctxmean', 'nn1'):
        out[f'D_{n}'] = float(np.mean((f - R[n]) ** 2))
    if boot:
        rng = np.random.default_rng(seed)
        bs = [stats(rng.integers(0, len(y), len(y))) for _ in range(boot)]
        out['S_lo'], out['S_hi'] = np.percentile([x[3] for x in bs], [2.5, 97.5]).tolist()
        out['rho_lo'], out['rho_hi'] = np.percentile([x[4] for x in bs], [2.5, 97.5]).tolist()
    return out


def load_results(folder):
    res = {}
    for name in sorted(os.listdir(folder)):
        m = re.match(r'probe_(.+)_step(\d+)\.jsonl$', name)
        if m:
            rows = [json.loads(l) for l in open(os.path.join(folder, name))]
            for k in sorted({r['k'] for r in rows}):
                res[(m.group(1), int(m.group(2)), k)] = [r for r in rows if r['k'] == k]
    return res


def paired_change(rows_a, rows_b, boot=2000, seed=0):
    """S(b) - S(a) on the same tasks, with a 95% paired bootstrap interval."""
    ida = {r['id']: r for r in rows_a}; common = [r['id'] for r in rows_b if r['id'] in ida]
    A = [ida[i] for i in common]; B = {r['id']: r for r in rows_b}; B = [B[i] for i in common]
    d0 = summarize_rows(B)['S'] - summarize_rows(A)['S']
    rng = np.random.default_rng(seed); ds = []
    for _ in range(boot):
        idx = rng.integers(0, len(common), len(common))
        ds.append(summarize_rows([B[j] for j in idx])['S'] - summarize_rows([A[j] for j in idx])['S'])
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return d0, lo, hi


def cmd_summarize(args):
    res = load_results(args.out)
    if not res:
        sys.exit(f'no results in {args.out}')
    order = [s for s in SIZES if any(m == s for m, _, _ in res)]
    table = []
    for size in order:
        for k in sorted({k for m, _, k in res if m == size}):
            steps = sorted(s for m, s, kk in res if m == size and kk == k)
            for step in steps:
                s = summarize_rows(res[(size, step, k)], boot=1000)
                table.append({'model': size, 'k': k, 'step': step, **s})
            # persistence: final checkpoint against the checkpoint with the highest S
            best = max(steps, key=lambda st: summarize_rows(res[(size, st, k)])['S'])
            last = steps[-1]
            if best != last:
                d, lo, hi = paired_change(res[(size, best, k)], res[(size, last, k)])
                print(f'{size} k={k}: S peaks at step {best}; final minus peak = {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})')
            else:
                print(f'{size} k={k}: S is highest at the final checkpoint ({last})')
    keys = ['model', 'k', 'step', 'n', 'S', 'S_lo', 'S_hi', 'rho', 'rho_lo', 'rho_hi', 'mse_model',
            'mse_bayes', 'mse_ols', 'mse_ctxmean', 'mse_nn1', 'D_bayes', 'D_ctxmean', 'D_nn1',
            'parse_rate', 'clip_rate']
    with open(os.path.join(args.out, 'summary.csv'), 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in table:
            f.write(','.join(str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in keys) + '\n')
    print(f'\n{"model":6s}{"k":>4s}{"step":>8s}{"S":>8s}{"95% CI":>18s}{"rho":>8s}{"parsed":>8s}')
    for r in table:
        print(f'{r["model"]:6s}{r["k"]:4d}{r["step"]:8d}{r["S"]:8.3f}   [{r["S_lo"]:+.3f}, {r["S_hi"]:+.3f}]'
              f'{r["rho"]:8.2f}{r["parse_rate"]:8.2f}')
    try:
        plot(table, os.path.join(args.out, 'probe_summary.png'))
    except ImportError:
        print('matplotlib not installed; skipped the figure')
    print(f'\nwrote {args.out}/summary.csv')


def plot(table, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shades = ['#9ecae1', '#4292c6', '#2171b5', '#08519c', '#08306b']
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, size in enumerate(dict.fromkeys(r['model'] for r in table)):
        for k in sorted({r['k'] for r in table if r['model'] == size}):
            rr = [r for r in table if r['model'] == size and r['k'] == k]
            st = [r['step'] for r in rr]
            lab = f'Pythia-{size}' + (f', k = {k}' if len({r["k"] for r in table}) > 1 else '')
            c = shades[min(i + 1, len(shades) - 1)]
            a.plot(st, [r['S'] for r in rr], marker='o', ms=4, color=c, label=lab)
            a.fill_between(st, [r['S_lo'] for r in rr], [r['S_hi'] for r in rr], color=c, alpha=0.15, lw=0)
            b.plot(st, [r['rho'] for r in rr], marker='o', ms=4, color=c, label=lab)
    for ax in (a, b):
        ax.set_xscale('log'); ax.set_xlabel('pretraining step')
        ax.spines[['top', 'right']].set_visible(False)
    a.axhline(0, color='#77766f', lw=0.9); a.axhline(1, color='#77766f', lw=0.9, ls=(0, (5, 2.5)))
    a.set_ylabel('in-context score S'); a.legend(frameon=False, fontsize=8)
    b.set_yscale('log'); b.axhline(1, color='#77766f', lw=0.9, ls=(0, (5, 2.5)))
    b.set_ylabel('error ratio to Bayes')
    fig.tight_layout(); fig.savefig(path, dpi=200)
    print(f'wrote {path}')


# ------------------------------------------------------------------------------------------ selftest
def cmd_selftest(args):
    """Runs the whole pipeline on a tiny untrained model and a tokenizer built on the spot, so the
    installation can be checked without downloading anything."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast
    tasks = make_tasks(24, 10, 0)
    corpus = [prompt(t) for t in tasks] * 4
    raw = Tokenizer(models.BPE(unk_token='[UNK]'))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=200, special_tokens=['[UNK]', '<|endoftext|>']))
    from tokenizers import decoders
    raw.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, eos_token='<|endoftext|>', pad_token='<|endoftext|>')
    tok.padding_side = 'left'
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(vocab_size=len(tok), hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                        intermediate_size=128, max_position_embeddings=512)
    model = GPTNeoXForCausalLM(cfg).eval()
    rows = evaluate(model, tok, tasks, 'cpu', batch=8)
    s = summarize_rows(rows, boot=200)
    print(f'selftest: {len(rows)} prompts, parse rate {s["parse_rate"]:.2f}, S = {s["S"]:+.3f}, '
          f'MSE(bayes) = {s["mse_bayes"]:.3f}, MSE(ctxmean) = {s["mse_ctxmean"]:.1f}')
    print('example prompt:\n' + prompt(tasks[0])[-60:])
    print('the pipeline works; an untrained model should score near S = 0 or below')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('--models', nargs='+', default=['70m', '160m'], choices=list(SIZES))
    r.add_argument('--steps', nargs='+', type=int, default=DEFAULT_STEPS)
    r.add_argument('--k', nargs='+', type=int, default=[10], help='demonstrations per prompt')
    r.add_argument('--n-tasks', type=int, default=200)
    r.add_argument('--seed', type=int, default=2026)
    r.add_argument('--batch', type=int, default=16)
    r.add_argument('--threads', type=int, default=0)
    r.add_argument('--cpu', action='store_true', help='use the CPU even if a GPU is present')
    r.add_argument('--keep', action='store_true', help='keep downloaded checkpoints')
    r.add_argument('--out', default='results')
    r.add_argument('--cache', default='hf_cache')
    s = sub.add_parser('summarize')
    s.add_argument('--out', default='results')
    sub.add_parser('selftest')
    args = ap.parse_args()
    {'run': cmd_run, 'summarize': cmd_summarize, 'selftest': cmd_selftest}[args.cmd](args)


if __name__ == '__main__':
    main()
