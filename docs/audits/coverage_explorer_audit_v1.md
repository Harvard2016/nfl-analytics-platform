# Coverage explorer audit, version 1 (2026-10-04)

Preserved summary of the audit run at commit `f78c522`, before the presentation fixes in `4846cca`.

**Finding.** Data, labels, saved predictions and on-screen values were consistent end to end. The explorer looked worse than
the published table because (1) 60 of its 120 exported plays were deliberately chosen snap-frame logistic errors, (2) it showed
logistic regression while the published table was boosted trees, and (3) it opened on the snap-frame window with a 0.75 cutoff.

**Subset vs full test (logistic, snap frame).** Disagreement 50.0% vs 18.5%; abstention 45.0% vs 30.0%; man share 41.7% vs 27.5%.

**Full locked test, 3,178 plays (873 man, 2,305 zone), boosted trees.**

| Window | Accuracy | Man recall | Zone recall | Cutoff | Accepted | Accuracy on accepted | Man correct and accepted / 873 |
|---|---|---|---|---|---|---|---|
| Snap | 0.8310 | 0.566 | 0.932 | 0.75 | 2,382 (75.0%) | 0.9060 | 302 (34.6%) |
| +1.0 s | 0.8697 | 0.670 | 0.945 | 0.65 | 2,828 (89.0%) | 0.8999 | 492 (56.4%) |
| +1.5 s | 0.8883 | 0.730 | 0.948 | 0.60 | 2,992 (94.1%) | 0.9111 | 583 (66.8%) |

**Checks that passed.** Class order (`classes_ = [0, 1]`, column 1 is man); calibrated sigmoid on weeks 13-14; joins on game and
play id together; 720 exported predictions matched recomputation to 0.00005; labels matched the raw file for all exported plays;
3 unlabelled plays (week 12) account for 14,108 vs 14,105.

**Fixes made afterwards (`4846cca`).** Prediction-blind random sample in the main explorer; separate error view per model and
window; model selector defaulting to boosted trees at +1.5 s; links carry model and window; explanations matched to the model;
one status definition. Models, calibration, cutoffs and evaluation results were not changed.

Exact metrics and hashes are in `reports/v1/baseline_manifest.json`.
