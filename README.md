# When Does In-Context Learning Emerge, and Does It Endure?

Companion repository for the study *When Does In-Context Learning Emerge, and Does It Endure?
Evidence from Small Transformers* by Maikel Leon (Department of Business Technology, University
of Miami Business School).

The study trains small decoder-only transformers on synthetic regression tasks drawn from a finite
pool of M tasks, a setting in which the Bayes-optimal memorizing predictor (the discrete minimum
mean-square error predictor, dMMSE) and the Bayes-optimal generalizing predictor (ridge regression,
or the family's exact Bayes predictor) have closed forms. It maps where behavior moves from the
first toward the second as the pool grows, across four architectures from 26,497 to 2,230,849
parameters, three task families (dense linear, sparse linear, and nonlinear random-feature
regression), three input dimensions, extended training, curriculum controls, and independent pool
draws, and it probes the attention heads that each solution relies on.

This repository holds the code that produced every number, table, and figure in the manuscript;
the raw evaluation log of each of the 212 training runs; the final weights of the 62 runs added in
the revision; the checkpoints of the revision's instrumented runs; and the four figures.

## Contents

```
code/
  icl_pilot.py        training script: task families, exact reference predictors (ridge,
                      dMMSE, sparse support enumeration, feature-space ridge), transformer,
                      dimension curriculum, training loop, and evaluation
  sweep.sh            phase-diagram campaign, Small and Medium rows
  sweep3.sh           phase-diagram campaign: seeds, fine grid, Large row
  sweep4.sh           extended-training campaign (100,000 steps)
  sweep5.sh           instrumented runs and probes, dimension sweep, preregistered sparse family,
                      Large-row seeds
  sweep6.sh           curriculum-free instrumented controls, second and third pool draws
  sweep7.sh           signal-matched sparse family, dimension-sweep seeds, learning-rate control,
                      curriculum-free anchor
  sweep8.sh           XLarge architecture, remaining grid cells and seeds
  queue_rev1.txt      the 62 revision runs, one command per line
  runq.py             resumable two-worker job queue that ran queue_rev1.txt
  mech_probe.py       checkpoint probes of the first instrumented campaign
  mech_probe2.py      checkpoint probes of the revision: any architecture or pool, zero- and
                      mean-ablation measured on fresh-task and pool contexts
  prereg.md           preregistration of the boundary-law test, recorded before the held-out runs
  rev_analysis.py     aggregates quoted in the text: phase-diagram cells, fine-resolution crossings
                      with pool bootstrap, prediction-level distances, architecture controls,
                      nonlinear family, dimension sweeps, run inventory
  rev_infer.py        analyses on saved weights: bootstrap and paired standard errors,
                      demonstration diversity at inference, sparse-family reference distances
  rev_distinct.py     distinguishability of pool tasks (normalized dMMSE posterior entropy)
  rev_mech.py         mechanism summaries across seeds and architectures
  rev_figs.py         Figures 1 to 4
  rev_tables.py       Tables 2 and 4 (LaTeX)
  requirements.txt    pinned package versions
  results/            logs (*.jsonl), analysis outputs (rev_*.json), final weights (final_*.pt),
                      and checkpoints (ckpt_*.pt)
  figs/               Figures 1 to 4 as they appear in the manuscript
```

## Requirements

Python 3.13 and the packages pinned in `code/requirements.txt` (PyTorch, NumPy, Matplotlib):

```
pip install -r code/requirements.txt
```

A CPU is sufficient for everything, including training. The CPU-only build of PyTorch is enough:
`pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

## Reproducing the figures, tables, and quoted numbers

These steps read the shipped logs and weights and need no training. Run them from `code/`:

```
python3 rev_figs.py fig1 fig2 fig3 fig4   # Figures 1 to 4, written to figs/
python3 rev_tables.py                      # Table 2 and Table 4 (LaTeX), printed
python3 rev_analysis.py                    # every aggregate quoted in the text, printed by section
python3 rev_mech.py                        # mechanism summaries (results/rev_mech_summary.json)
python3 rev_infer.py boot                  # bootstrap standard errors (results/rev_boot.json)
python3 rev_infer.py boot_paired           # paired standard errors (results/rev_boot_paired.json)
python3 rev_infer.py demo                  # demonstration diversity (results/rev_demo.json)
python3 rev_infer.py demo_summary          # the summary of rev_demo.json quoted in the text
python3 rev_infer.py sparse                # sparse-family distances (results/rev_sparse.json)
python3 rev_distinct.py                    # distinguishability (results/rev_distinct.json)
```

Each command finishes in seconds to a few minutes on a laptop CPU. `rev_analysis.py table` (or
`fine`, `pred`, `arch`, `nl`, `dsweep`, `inventory`) prints a single section.

The mechanistic probes rerun from the shipped checkpoints of the revision's instrumented runs, for
example:

```
python3 mech_probe2.py --tag mech_m_M16_s1 --M 16                                # Medium
python3 mech_probe2.py --tag mech_s_M2 --M 2 --layers 2 --dmodel 32 --heads 2   # Small
```

Each probe writes `results/mech2_<tag>.jsonl` in a few minutes. Checkpoints are included for
`mech_m_M{2,16,2048}_s{1,2}` and `mech_s_M{2,16,2048}`. The probe outputs of the first
instrumented campaign (`mech_mech_*`, `mech_mechnc_*`) are included, but their checkpoints are not;
the corresponding runs in `sweep5.sh` and `sweep6.sh` regenerate them.

## Retraining

Every run can be retrained from its command. The commands of the revision runs are listed in
`queue_rev1.txt`, and those of the earlier campaigns are in the sweep scripts. Training writes
`<out>/<tag>.jsonl` and overwrites an existing log of the same tag, so retrain into a separate
folder:

```
python3 icl_pilot.py --steps 20000 --eval-every 2000 --curr --auto-resume --save-final \
    --M 12 --seed 0 --tag m_M12_s0 --out retrained
```

A Medium run takes about 10 minutes on two CPU cores. To rerun a whole campaign, copy `code/`
without its `results/` folder, because the sweep scripts skip every run whose log is complete,
and run the scripts from the copy (`bash sweep.sh`, and so on). The revision campaign runs with two
workers:

```
python3 runq.py 1 queue_rev1.txt & python3 runq.py 2 queue_rev1.txt &
```

All randomness is seeded, and the evaluation sets are regenerated from fixed seeds. The revision
runs used one thread per run (`ICL_THREADS=1`, set by `runq.py`); the earlier campaigns used two.
With the same software versions and thread count, retraining reproduces a log exactly apart from
its wall-clock field (`secs`). Other hardware or library versions can change the last digits.

## Verification

On a clean copy of this repository, with the pinned versions on a Linux x86 CPU, the commands
above regenerate the four figures, the six analysis files in `results/`, and the probe output of
`mech_s_M2` byte for byte, and they print Tables 2 and 4 exactly as they appear in the manuscript.
Two runs retrained from scratch with one thread each, `s_M6_s0` (Small) and `m_M12_s0` (Medium),
reproduced every logged value and their final weights bit for bit. The logs of the 62 revision runs
record the PyTorch version (2.15.0); the logs of the earlier campaigns do not.

## Log format

Each `results/<tag>.jsonl` file holds one JSON record per line. The first record (`"type": "meta"`)
gives the configuration: architecture, pool size `M` (0 denotes an unlimited pool, in which every
context uses a fresh task), seed, family, input dimension `D`, context length `K`, noise level,
curriculum, pool seed base, and parameter count. Each later record (`"type": "eval"`) reports, at
one evaluation step, the training loss and, on the shared fresh-task (`_fresh`) and pool
(`_pool`) evaluation sets of 512 contexts, the mean-square errors of the model, ridge, and dMMSE
(`mse_*`) and the order parameter (`order_*`). Runs from the revision also report the squared
prediction distances from the model to ridge (`Dr_*`), from the model to dMMSE (`Dg_*`), and
between the two references (`Drg_*`).

Run tags encode the configuration: the prefix gives the architecture (`s_` Small, `m_` Medium,
`l_` Large, `xl_` XLarge, `wd_` Wide, `dp_` Deep), the campaign (`poolB_` and `poolC_` for the
second and third pool draws, `d2_` and `d6_` for input dimensions 2 and 6, `sp_` and `spm_` for the
preregistered and signal-matched sparse families, `nl_`, `nlB_`, and `nlC_` for the nonlinear family
and its pool draws, `mech_` and `mechnc_` for instrumented runs with and without the curriculum,
`nc_` for the curriculum-free anchor, `rr_` for reruns that save final weights), followed by
`M<pool size>` and the seed (`_s0` to `_s2`). The suffix `_long` marks 100,000-step runs and
`_lr3e4` the learning-rate control. Files starting with `mech_mech_`, `mech_mechnc_`, and `mech2_`
are probe outputs rather than training runs.

## Citation

Please cite the paper when using this code. `CITATION.cff` gives the citation for the software
itself.

## License

The code is released under the MIT License (`LICENSE`).
