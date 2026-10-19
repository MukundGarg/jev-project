# Architecture

## 1. Components and responsibilities

| Component | Input | Output | Must NOT |
|---|---|---|---|
| Vision Context Generator | one image | structured context (JSON) | decide relevance to the user's query |
| JEV Adapter | one structured context + query | relevance probability in [0,1] | see or receive image pixels |
| Ranker | list of JEV outputs | ordered results, best match or "none" | re-inspect images |
| Pipeline | images + query | final response | contain model-specific logic |
| UI (Gradio) | uploads + query | rendered response | contain business logic |

Suggested layout:

```
contextselect/
  vision/        base.py (interface), <backend>.py, prompts/context_v1.txt
  jev/           base.py (interface), adapter.py
  schema.py      context schema + validation
  ranker.py
  pipeline.py
  cache.py       on-disk cache keyed by image hash + prompt version + model id
app.py           Gradio UI
eval/            datasets manifest, run_eval.py, baselines.py, results/
docs/
tests/
```

## 2. Interfaces (design level)

- `ContextGenerator.generate(image) -> Context`
- `Classifier.score(context, query) -> Judgement{score}` where `score` is JEV's relevance probability in [0,1]. No free-text rationale is requested. The adapter remains modular behind this interface.
- `rank(items, confidence_threshold=0.5, ambiguity_margin=0.05) -> RankedResult`; results are sorted descending by score, with stable input order for ties. Scores below the confidence threshold yield `no_confident_match`; a top-two score gap less than or equal to the ambiguity margin yields `ambiguous`; otherwise the top image is `best_match`.

Both models are behind these interfaces so baselines and ablations in evaluation reuse the same pipeline.

The TypeSafe Jev adapter uses `POST https://api.typesafe.ai/v1/systemone`. Set the `Authorization` header using the API token loaded from the `TYPESAFE_API_KEY` environment variable; never hardcode or commit it. For each image, send that image's structured context as `state` and a `questions` object with a `"match"` question of type `"noul"`. Its instructions ask whether the context matches the user's target concept/query. Read `answers.match.noul` as the relevance probability in [0,1]. Do not send image pixels or request a free-text rationale. Adapter behavior must follow the confirmed TypeSafe API contract.

## 3. Semantic context schema (v1)

Structured JSON instead of free-form prose. Reasons: consistent fields across images, easier validation, enables field-level ablations, easier for JEV to consume.

```
subject:          most specific identification, plus one alternative if uncertain
category:         broader class (e.g. tool, vehicle, food, animal, electronics)
appearance:       shape, color, material, texture, size cues
distinctive_features: parts or markings that separate it from lookalikes
function:         what it is used for or does
state_condition:  intact / damaged / opened / in use, with visible evidence
action_context:   what is happening, who or what interacts with it
environment:      setting and scene
visible_text:     any legible text, logos or labels (verbatim)
related_concepts: similar or commonly confused items, parent categories
uncertainty:      what is unclear, occluded, or low-confidence
```

Rules in the prompt: describe only what is visible; mark inferences as inferences; no poetic language; do not guess when unclear. Prompts are versioned files (`context_v1.txt`).

`state_condition`, `visible_text` and `uncertainty` are additions to your list, justified by your own example queries (damaged product, product category).

## 4. Data flow

1. Upload -> validate -> hash.
2. Cache lookup by (hash, prompt version, vision model id). Miss -> call vision backend at temperature 0 -> parse JSON -> validate against schema -> one retry on failure -> cache.
3. For each image, call Jev separately with only that image's structured context as `state` and the user's query in the `"match"` `noul` question instructions.
4. Read `answers.match.noul` as the image's relevance probability in [0,1].
5. Ranker orders by score; if top score < threshold, result is "no confident match". Ties and near-ties (margin below a small delta) are surfaced as "ambiguous" with the tied images listed.
6. Response includes everything needed to explain the result.

## 5. Model responsibilities

- **Vision model:** perception and description. Query-agnostic, so each image is processed once and context is reusable across queries.
- **JEV:** reasoning over text evidence against a concept. This is the only component that makes the relevance decision; the ranker compares its returned probabilities.
- **Honest caveat:** if the context says "subject: dog", the decision is trivial, and the vision model has effectively done the classification. This is addressed by the identification-withheld ablation in EVALUATION.md. The README must state this plainly rather than hide it.

## 6. Technology choices

- Python 3.11+, Pydantic for the schema, Gradio for UI (one file, shareable demo), pytest, `requirements.txt` or `pyproject.toml` with pinned versions.
- VLM: Gemini 2.5 Flash via hosted API. Hosted APIs are acceptable; the interface keeps the backend swappable.
- Maximum images per query: 8.
- JEV: official TypeSafe Jev API, `POST https://api.typesafe.ai/v1/systemone`; set the `Authorization` header using a token loaded from `TYPESAFE_API_KEY`. Use the `noul` question type for per-image relevance scoring. The adapter receives structured context and query only, and reads the probability from `answers.match.noul`.

## 7. Error handling

- Invalid or oversized image: reject with message.
- Vision backend timeout or error: retry once, then mark that image "failed" and continue with the rest.
- Malformed JSON: one repair retry, then mark failed.
- All images failed: clear error, no fabricated result.
- JEV unavailable: surface the error, do not fall back silently to the vision model.
