# Preregistration: boundary law and held-out family test

Recorded 2026-07-22, before any held-out-family training run was launched.
This file is committed to the released repository unchanged.

## Fitted law (training data: linear family only)

Using the completed linear-family phase diagrams at d = 4 (all sizes) and the
d-sweep at d = 2 and d = 6 (medium model, 20k steps, batch 64, curriculum,
identical protocol), we estimate the boundary M*(d) as the interval in which
the order parameter crosses zero, and fit the one-parameter law

    M*(d) = alpha * d.

Expectation from the d = 4 data alone: alpha in [2, 4].

## Held-out family

Sparse linear regression: d = 4, s = 2 active coordinates chosen uniformly,
active weights N(0, 1), noise sigma = 0.25, K = 12, medium model, 20k steps,
seeds {0, 1}, diversity grid M in {2, 4, 8, 16, 32, 128}. The ICL pole is the
exact Bayes predictor under the sparse prior (support enumeration); the
memorization pole is dMMSE over the pool. Neither the grid nor any
hyperparameter will be changed after seeing results.

## Preregistered hypotheses

H1 (law): the zero crossings at d = 2 and d = 6 are consistent with
M*(d) = alpha * d for a single alpha in [2, 4]; concretely, the d = 2 crossing
lies in [4, 16] and the d = 6 crossing lies in [12, 48], i.e., within a factor
of two of alpha * d for the alpha fitted at d = 4.

H2 (held-out family): the sparse family has effective dimensionality between
s = 2 and d = 4; the fitted law therefore predicts its boundary within
[alpha * 2, alpha * 4]. With the d = 4 estimate alpha in [2, 4], the
preregistered prediction is that the sparse-family order parameter crosses
zero within M in [4, 16], and in particular that the sparse crossing does not
occur later than the linear d = 4 crossing.

## Decision rule

For each family and d, the crossing interval is the pair of adjacent grid
points between which the seed-mean order parameter changes sign. H1 or H2
passes if the observed interval overlaps the preregistered interval, and fails
otherwise. Failures will be reported as failures.
