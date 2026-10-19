# Design Review: Critique of the Original Idea

Format per item: **Original** / **Recommendation** / **Reason**.

## 1. JEV API contract (current)

- **Original:** JEV is "the final classification model" that receives context and classifies.
- **Current interface:** TypeSafe Jev through OpenRouter's Decisions API, `POST https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-router`. Set the `Authorization` header using a key loaded only from `OPENROUTER_API_KEY`; never hardcode or commit it. `.env` is excluded through `.gitignore`.
- **Request:** One call per image. Send that image's structured semantic context as `state`; use `questions.match.type = "noul"` and ask in `questions.match.instructions` whether the context matches the user's target concept/query.
- **Response:** Read `answers.match.noul` as the relevance probability in [0,1]. Rank image contexts by their returned probabilities. Use thresholds for no-match/ambiguity in the ranker later.
- **Privacy and output:** Jev receives structured context and the query only, never image pixels. Do not require or request a free-text rationale; Jev is used as a typed decision model.
- **Adapter contract:** `Classifier.score(context, query) -> {score}`. Keep the API-specific adapter modular; no adapter implementation is part of Phase 0.

## 2. Terminology and framing

- **Original:** "Vision-to-Context Classification".
- **Recommendation:** Frame it as *query-conditioned image selection through a semantic text bottleneck*, and describe the study as an *information-preservation evaluation of vision-to-text representations*.
- **Reason:** "Classification" implies fixed labels, which contradicts your generalization goal. The new framing is accurate and gives you a defensible research question for interviews.

## 3. Binary classification vs ranking for multi-image input

- **Original:** Classify each image, compare results, pick the match.
- **Recommendation:** Per-image relevance scoring, then rank. Add an explicit "no confident match" threshold and an "ambiguous" outcome for near-ties.
- **Reason:** Binary labels break when zero or several images match, and cannot rank among several plausible candidates. Scores allow all three cases. Requirement: JEV scores must be comparable across images, which should be checked in evaluation.

## 4. Query-agnostic vs query-aware context

- **Original:** Not specified; example implies query-agnostic.
- **Recommendation:** Keep context **query-agnostic** (generated once per image, cached).
- **Reason:** It matches your stated architecture, makes the bottleneck test clean (the VLM cannot tailor evidence to the answer), and makes caching trivial. A query-aware variant would be a legitimate extension but is out of scope by default.

## 5. Structured context instead of free-form text

- **Original:** Free-form semantic paragraph, structure optional.
- **Recommendation:** JSON schema (see ARCHITECTURE.md) with added fields: `category`, `state_condition`, `visible_text`, `uncertainty`. Drop `behavior` as a standalone field (merged into `action_context`) and cut poetic or speculative content.
- **Reason:** Consistency, validation, and field-level ablations. The additions are required by your own example queries (damaged product, product category). The `uncertainty` field gives JEV a way to discount weak evidence.

## 6. Label leakage risk (most important ML issue)

- **Original:** Vision model must not "directly perform classification".
- **Recommendation:** Accept that useful context will often name the subject, but measure the effect: include the S5 ablation where `subject` and `category` are withheld, and report both results.
- **Reason:** If the context says "dog", JEV's job is trivial and the pipeline is just "VLM classification with extra steps". Banning identification outright would hurt generalization (a "laptop" description without the word is a weak signal). Measuring it honestly turns a weakness into your strongest interview answer. Also, in the cat example, "resembles tigers" is a related concept, not evidence; the prompt must keep inferences labeled.

## 7. Evaluation is mandatory, not optional

- **Original:** "Only include if appropriate and manageable".
- **Recommendation:** Include it, scoped as in EVALUATION.md (N-way selection, 6 systems, ~few hundred images).
- **Reason:** Without it the project is API plumbing, and you could not answer "how do you know the context is good enough?". It is the main source of resume value.

## 8. Stop condition adjusted

- **Original:** Completion = working system + docs.
- **Recommendation:** Add: evaluation run once, results and failure analysis in README.
- **Reason:** A portfolio ML project without a result table is incomplete.

## 9. Things deliberately NOT added

Vector DB, RAG, agents, auth, databases, training, dashboards. A file cache keyed by image hash is the only persistence. Gradio is chosen over a custom frontend for effort/benefit.

## 10. Expectation to set honestly

The pipeline will likely not beat a strong direct VLM on plain recognition. Its defensible advantages are interpretability (readable evidence), modularity (swap the text classifier without touching vision), and reusability (context generated once, queried many times). The README should say this directly.

## 11. Documentation set chosen

| File | Why it exists |
|---|---|
| PROJECT_SPEC.md | purpose, requirements, scope, plan, completion criteria |
| ARCHITECTURE.md | components, schema, data flow, error handling |
| EVALUATION.md | hypothesis, datasets, baselines, metrics |
| DESIGN_REVIEW.md | this critique and open decisions |
| README.md | written at the end, not now (needs real results) |

Not created: separate requirements, user-story, API, or changelog docs. They would duplicate these files at this stage.

## 12. Confirmed Phase 0 decisions

**Confirmed:** Gemini 2.5 Flash is the Vision Context Generator; hosted APIs are acceptable; the maximum is 8 images per query; TypeSafe Jev is called once per image through OpenRouter's Decisions API using model `typesafe/jev-router` and its `noul` question type; its relevance probability is read from `answers.match.noul`; the API key is loaded from `OPENROUTER_API_KEY` and sent in the `Authorization` header; and the adapter interface is `Classifier.score(context, query) -> {score}`. Jev never receives image pixels, and no free-text rationale is required.

**Phase 0 is complete.** The JEV adapter and all other application code remain unimplemented; Phase 1 has not begun.
