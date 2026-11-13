# Probe of pretrained checkpoints

This folder holds the probe of pretrained language models reported in Section 9.2 and Figure 6
of the paper. It evaluates in-context regression across the pretraining checkpoints of the Pythia
suite (Biderman et al., 2023, ICML), whose models of different sizes were trained on the same
deduplicated corpus in the same order and released with intermediate checkpoints, so that model
size and training time vary while the data are held fixed.

## What it measures

Each prompt shows k = 10 noisy (x, y) pairs from a random linear function, written as text
("Input: 3.42" followed by "Output: 41.93" on the next line), and asks for y at a new x. Tasks
draw y = a + w x + noise with a ~ N(30, 8^2), w ~ N(4, 1.5^2), x ~ U[0, 10], and noise sd 1,
with every number rounded to two decimals as shown. Because the prior is known, the
Bayes-optimal prediction (the posterior mean) is exact for every prompt. Greedy decoding
produces the answer, and its first number is the prediction; an answer without a number is
replaced by the mean of the demonstrated outputs, and a prediction farther outside the range of
the demonstrated outputs than the width of that range is clipped to that limit.

Two statistics follow the paper's measurement framework:

- the error ratio rho = MSE(model) / MSE(Bayes);
- an in-context score S = (MSE(context mean) - MSE(model)) / (MSE(context mean) - MSE(Bayes)),
  which is 0 for a model that only reproduces the average demonstrated output and 1 for the
  Bayes-optimal predictor.

The same 200 tasks (seed 2026) are shown to every checkpoint and model, so changes between
checkpoints are paired comparisons; intervals are 95% bootstrap intervals over tasks. Least
squares on the demonstrations, the mean of the demonstrated outputs, and the output of the
nearest demonstration are computed as further references.

## What it can and cannot show

The probe tests two behavioral implications of the paper in pretrained models: whether in-context
regression improves with model size (in the controlled setting, capacity lowers the error of the
generalizing solution) and whether, once formed, it persists through the rest of pretraining (in
the controlled setting, ICL persists when pretraining diversity lies well above the boundary). It
cannot test the diversity gate itself, because the pretraining data of released models cannot be
varied, and it cannot separate capacity from the ability to read and write numbers as text.

## Results included

`results/probe_<size>_step<step>.jsonl` holds one JSON record per prompt: model, checkpoint,
task id, k, the query target `yq`, the prediction `pred`, the first characters of the raw answer,
the flags `parsed` and `clipped`, and the four reference predictions (`ref_bayes`, `ref_ols`,
`ref_ctxmean`, `ref_nn1`). The shipped results cover all six sizes, Pythia-70M, -160M, -410M, -1B, -1.4B, and -2.8B
(deduped), at steps 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, and 143000, 200
prompts each. They were produced on Colab T4 GPUs in half precision, in two sessions (the two
largest sizes in the second).

## Reproducing the figure and the numbers

From `code/`, with NumPy and Matplotlib only (no GPU, no downloads):

```
python3 rev_figs.py fig6                                          # Figure 6, written to figs/
python3 llm_probe/llm_probe.py summarize --out llm_probe/results   # table, CSV, and figure
```

## Rerunning the probe

A GPU is recommended. On Google Colab, upload `llm_probe_colab.ipynb` (Pythia-70M to -1B) or
`llm_probe_colab_round2.ipynb` (Pythia-1.4B and -2.8B), choose a T4 GPU runtime, and run all
cells; the last cell downloads the results as a zip file. Locally:

```
pip install -r requirements.txt
python llm_probe.py selftest                              # no downloads, about a minute
python llm_probe.py run --models 70m 160m 410m 1b         # default checkpoints, k = 10
python llm_probe.py summarize
```

A run that stops can be restarted with the same command; it skips what is already done. Each
checkpoint is deleted after its evaluation unless `--keep` is given, so about 1 GB of free disk
(6 GB for Pythia-2.8B) is enough. The CPU also works (`--cpu`) but takes hours. Greedy decoding
in half precision on other hardware can change a small number of predictions.

## Files

- `llm_probe.py`: the probe (`selftest`, `run`, `summarize`)
- `llm_probe_colab.ipynb`, `llm_probe_colab_round2.ipynb`: the same code as Colab notebooks
- `requirements.txt`: Python packages
- `results/`: per-prompt outputs of the shipped run
