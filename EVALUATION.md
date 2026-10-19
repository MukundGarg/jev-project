# Evaluation Plan

## 1. What is being tested

Primary hypothesis: structured semantic context preserves enough information for a text-only classifier to select the correct image.

The expected outcome is NOT necessarily "beats direct classification". Direct vision models are usually stronger on pure recognition. The deliverable is a measured comparison: how much accuracy is paid for modularity, interpretability and a swappable text classifier, and where the loss occurs.

## 2. Task format

**N-way selection:** N images (default 4), one target concept, exactly one correct image. This mirrors the product use case and gives a single clean metric.

Also include a small set of **no-match** trials (no image satisfies the query) to test the threshold.

## 3. Data (kept manageable)

- Tier A, standard recognition: a subset of a public labeled dataset (for example Oxford-IIIT Pets, or a few dozen ImageNet-style categories). Used to build N-way trials automatically.
- Tier B, generalization: ~100-200 hand-labeled images covering function-based and state-based queries (cutting tool, writing instrument, flying vehicle, food, damaged vs intact product). Queries are phrased as concepts and not class names.
- Tier C, hard negatives: lookalike pairs (e.g. wolf/dog, knife/screwdriver) to probe information loss.

Check the license of each dataset and list sources in the README. Record the exact image list in a manifest file for reproducibility. Avoid test images that appear in obvious public benchmarks if the VLM may have memorized them; note this limitation.

## 4. Systems compared

| ID | System | Purpose |
|---|---|---|
| S0 | Random | floor |
| S1 | Direct: CLIP-style zero-shot image-text matching | classic direct baseline |
| S2 | Direct: VLM asked the query with all images | strong direct baseline |
| S3 | Free-form caption -> JEV | does structure matter? |
| S4 | **Structured context -> JEV (proposed)** | main system |
| S5 | Structured context with `subject` and `category` withheld -> JEV | does description alone carry enough evidence? |

S5 is the key ML experiment: it separates "the VLM already named the answer" from "the description genuinely preserves discriminative evidence".

## 5. Metrics

- **Primary:** top-1 selection accuracy over N-way trials.
- Per-concept precision, recall, F1 for the single-image decision (score >= threshold) on Tier A/B.
- No-match trials: false-accept rate and correct-reject rate.
- Latency per image and per query (context generation vs JEV, reported separately); cost per query if using paid APIs.
- Score calibration (optional, only if cheap): do higher JEV scores correspond to higher accuracy? Simple reliability bins.
- Report counts of trials, and a confidence interval (bootstrap) on accuracy so small-sample differences are not overclaimed.

## 6. Failure analysis (required)

Manually categorize all errors of S4 into:
1. Perception error (VLM described something wrong)
2. Information loss (correct but too coarse to discriminate)
3. Reasoning error (JEV misjudged correct evidence)
4. Ambiguous or bad label

This breakdown is the most interview-relevant output of the project.

## 7. Reproducibility

Temperature 0, versioned prompts, pinned model identifiers, cached contexts stored with their hashes, one command (`python eval/run_eval.py`) regenerates the results table in `eval/results/`.

## 8. Limits of this evaluation

Small dataset, single VLM, English only, and results may not transfer to other domains. State this in the README.
