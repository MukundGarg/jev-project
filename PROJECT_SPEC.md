# Project Spec: Query-Conditioned Image Selection via Semantic Context

Working title: **ContextSelect** (rename freely).

## 1. Purpose

Test, and demonstrate, one concrete ML question:

> Can an image be converted into structured text that retains enough evidence for a downstream **text-only** classifier (JEV) to make the correct decision about the original image?

The system is a modular pipeline, not a new vision model. Its value is the pipeline plus an honest measurement of what the text bottleneck preserves and what it loses.

## 2. Problem statement

Given N >= 1 images and a free-text query ("Which image shows a cutting tool?"), identify which image(s) best satisfy the query, without a fixed list of classes.

## 3. Pipeline (summary)

```
Image -> [Vision Context Generator] -> structured context (JSON)
Context + Query -> [JEV] -> relevance probability
Scores across images -> [Ranker] -> best match (or "none")
```

Details in ARCHITECTURE.md.

## 4. Assumptions and confirmed decisions

- A1. **JEV is accessed through the official TypeSafe Jev API.** Send one image's structured context as `state` and a `"match"` question of type `"noul"` to `POST https://api.typesafe.ai/v1/systemone`; read `answers.match.noul` as the relevance probability in [0,1]. Set the `Authorization` header using the API token loaded from the `TYPESAFE_API_KEY` environment variable. Never hardcode or commit the key. JEV must not receive image pixels and no free-text rationale is required.
- A2. Use Gemini 2.5 Flash via hosted API as the frozen, off-the-shelf vision component. Hosted APIs are acceptable; no training from scratch.
- A3. Images are single photographs or product shots, not video, medical or satellite imagery.
- A4. English-only queries and context.
- A5. Maximum 8 images per query.

## 5. User flow

1. User uploads 1-8 images.
2. User types a query describing the target concept.
3. System generates structured context per image (cached by image hash).
4. JEV scores each image's context against the query in a separate API call, returning one relevance probability per image.
5. UI shows: ranked images with scores, best match highlighted (or "no confident match"), and an expandable view of each image's generated context.

## 6. Functional requirements

- FR1. Accept 1-N image uploads (jpg/png/webp), validate type and size.
- FR2. Generate structured semantic context per image using a swappable vision backend.
- FR3. Pass context (not the image) to JEV; JEV must never see pixels.
- FR4. Return each image's JEV relevance probability in [0,1].
- FR5. For N > 1, rank images and return the best match; return "no confident match" when top score is below a threshold.
- FR6. Open-vocabulary: no hardcoded class list anywhere in the pipeline code.
- FR7. Display generated context in the UI (explainability).
- FR8. CLI or script entry point to run the evaluation (see EVALUATION.md).

## 7. Non-functional requirements (only those that matter)

- Reproducibility: pinned dependencies, fixed prompts versioned in repo, fixed seeds/temperature 0 for evaluation, one-command evaluation run.
- Modularity: vision backend and JEV are behind small interfaces; swapping either needs no pipeline changes.
- Error handling: bad file, backend timeout, malformed model output (schema validation + one retry), all produce clear messages.
- Latency is measured and reported, not optimized.

## 8. Out of scope

Auth, accounts, databases, vector stores, RAG, agents, cloud infra, microservices, mobile, training or fine-tuning, dashboards. Caching is a plain on-disk file cache, not a database.

## 9. Constraints and limitations (to be stated in README)

- Text is a lossy bottleneck; fine-grained visual cues (texture, exact shape, small defects) may not survive.
- Quality depends on the VLM; hallucinated attributes propagate to JEV.
- Open-vocabulary quality depends on JEV's ability to reason about concepts it was not trained on (depends on A1).
- Multi-image scores are only meaningful if JEV's scores are comparable across inputs.

## 10. Development plan

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Confirm JEV interface and VLM choice | Finalized VLM, hosting, image cap, and Jev API contract are recorded in ARCHITECTURE.md |
| 1 | Context generator + JSON schema + validation | 20 varied images produce valid, sensible context |
| 2 | JEV adapter + single-image scoring | Query + context -> score works end-to-end in a script |
| 3 | Multi-image ranker + "none" handling | N-way selection works from CLI |
| 4 | Evaluation harness + baselines + results table | EVALUATION.md metrics produced from one command |
| 5 | Gradio UI | Full flow demonstrable in browser |
| 6 | README, limitations, results, failure analysis | Repo readable cold by a stranger |

## 11. Completion criteria

The project is complete when all hold:

1. Image upload works.
2. Structured context generation works.
3. Context (not the image) reaches JEV.
4. JEV scoring works.
5. Multi-image ranking returns the correct best match on demo cases, including a "none" case.
6. UI demonstrates the whole flow.
7. **The evaluation has been run once and its results and failure analysis are in the README.** (Added to your original list; see DESIGN_REVIEW.md item 8.)
8. README documents architecture, setup, results, and limitations.

When these are met, the status line is: "The agreed project scope is complete. Stop here unless you intentionally want to extend it."
