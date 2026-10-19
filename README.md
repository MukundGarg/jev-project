# ContextSelect

ContextSelect selects the image that best matches a free-text query by translating each image into structured semantic evidence and scoring that evidence with Jev.

## Motivation

Image selection is often more useful when the system can compare a query against more than a single visual label. ContextSelect explores a modular two-stage approach: a vision model describes what is visibly present, and a typed decision model judges whether that description matches the user's target.

## Architecture and data flow

For each submitted image, the pipeline performs:

1. Gemini 2.5 Flash reads the image and produces a query-agnostic `Context`.
2. The OpenRouter adapter sends that structured context and the user's query to TypeSafe Jev's `typesafe/jev-router` model as a `noul` question named `match`.
3. The adapter reads `answers.match.noul` as a probability from 0 to 1 and returns a `Judgement`.
4. The ranker sorts images by score, preserving original order for equal scores. It returns `best_match`, `no_confident_match`, or `ambiguous`.

Jev never receives image pixels. Gemini handles image perception; Jev receives only generated structured context and the text query. The semantic text bottleneck keeps the providers independently replaceable and makes the evidence sent to the decision stage inspectable. It can also discard useful visual details, so this design is not expected to outperform a direct vision model on every recognition task.

### Context schema

Each image context contains:

- `subject`, `category`: concise identification and broad type.
- `appearance`: visible overall appearance.
- `distinctive_features`: list of visible distinguishing details.
- `function`: apparent or cautiously inferred function.
- `state_condition`: visible state, condition, or damage.
- `action_context`: visible action or interaction.
- `environment`: visible surroundings.
- `visible_text`: text that is actually legible in the image.
- `related_concepts`: other useful concepts grounded in visible evidence.
- `uncertainty`: ambiguities or limits of what can be determined.

The prompt is query-agnostic: it must not tailor context to the user's target. A key limitation is identity leakage: if Gemini writes an object's identity directly into `subject` or `category`, Jev's classification becomes easier and may partly reflect that label rather than richer visual evidence. The evaluation design includes a `structured_no_identity` ablation hook for investigating this effect; that baseline is currently an unimplemented extension point.

### Multi-image decisions

The pipeline accepts 1–8 images and evaluates each independently. Default ranking settings are confidence threshold `0.5` and ambiguity margin `0.05`. A top score below the threshold produces `no_confident_match`; otherwise, if the top two scores differ by no more than the ambiguity margin, the result is `ambiguous`; otherwise it is `best_match`. Only `best_match` has a best index.

## Setup

Requirements: Python 3.10+ and API access to Gemini and TypeSafe Jev.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Set the two values in the ignored `.env` file, then export them into the current shell before starting the app or evaluation:

```bash
set -a
source .env
set +a
```

The application reads credentials only from environment variables:

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Gemini 2.5 Flash context generation |
| `OPENROUTER_API_KEY` | TypeSafe Jev relevance scoring through OpenRouter |

Do not commit `.env` or put credentials in source code, manifests, or logs. `.env.example` contains empty placeholders only.

## Run the application

```bash
source .venv/bin/activate
set -a && source .env && set +a
python app.py
```

Open the local Gradio URL printed by the process. Upload 1–8 images, enter a query, and submit. The app displays the final status, ranked images and Jev scores, and each image's generated context. Missing credentials and provider errors are surfaced without displaying secret values.

The demo limits each client IP to 5 submissions per rolling minute and 50 submissions per UTC day. One submission counts once regardless of how many images it contains. These limits are held in memory, are not authentication, and reset when the hosted app restarts.

## Evaluation

The evaluation harness accepts JSON manifests. `default_n` defaults to 4 and must match the number of images in every trial. Each trial has an `id`, a non-empty `query`, four (or configured N) images with unique `id` and local `path`, and `correct_index` (zero-based) or `null` for a no-match trial. Paths are resolved relative to the manifest file. The example at `eval/examples/demo_manifest.json` documents the shape but references image files that are not included.

The primary metric is top-1 accuracy on normal trials. The harness also reports no-match correct-reject rate, false-accept rate, average end-to-end latency per trial, and trial count. It can run the proposed structured-context → Jev system and a deterministic seeded random baseline. Hooks for direct CLIP-style matching, direct VLM matching, caption → Jev, and the no-identity ablation are documented but not implemented.

To run an evaluation after supplying local images:

1. Add the images you want to evaluate under `eval/examples/images/` (or another local directory).
2. Edit `eval/examples/demo_manifest.json` so each path points to an existing image. Use exactly `default_n` images in every trial. Set `correct_index` to the correct image's zero-based position for normal trials; use `null` only when none of the images match.
3. Export both API keys as described above.
4. Run:

   ```bash
   python -m eval.run_eval \
     --manifest eval/examples/demo_manifest.json \
     --system structured_context_jev
   ```

The runner writes machine-readable JSON and CSV under `eval/results/`. API usage is one Gemini context-generation request and one Jev scoring request per image in each trial.

### Real evaluation results from this run

| Measure | Result |
| --- | --- |
| Completed real structured-context → Jev trials | 0 |
| Top-1 accuracy | Not available |
| No-match result | Not tested |
| Average latency | Not available |

No real provider calls were made: `GEMINI_API_KEY` and `OPENROUTER_API_KEY` were not configured in the execution environment, and no local image files were present. The example manifest's image paths are placeholders, not evaluation data. No real evaluation performance is claimed.

## Limitations and failure modes

- Direct vision models may outperform this semantic-context pipeline on pure recognition because the context-generation stage can omit or misidentify details.
- Identity in `subject` or `category` can leak the answer into Jev's input; evaluate the no-identity ablation before drawing conclusions about the value of other context fields.
- Scores are provider outputs, not calibrated probabilities of real-world correctness. The ranking thresholds are simple configurable defaults.
- Gemini or Jev configuration, network, response-format, or schema errors stop processing and are surfaced; the pipeline does not silently substitute a result.
- Real accuracy, no-match performance, and latency remain unmeasured until real API credentials and a labeled local image manifest are supplied.
- Evaluation results depend on the selected image set, queries, ground-truth indices, provider versions, and ranking thresholds.

## Tests

Run the mock-only suite with:

```bash
pytest -q
```

Provider tests use mocks and do not make live Gemini or Jev calls.

## Repository structure

```text
contextselect/
  vision/       Context generator interface, Gemini backend, and prompt
  jev/          Classifier interface and TypeSafe Jev adapter
  schema.py     Context and Judgement data models
  pipeline.py   Injected single- and multi-image orchestration
  ranker.py     Stable ranking and outcome selection
  cache.py      Cache extension interface
eval/
  manifest.py   Manifest schema and path resolution
  baselines.py  Evaluation baseline interfaces and implementations
  run_eval.py   Runner, metrics, and JSON/CSV output
tests/          Mock-only unit and integration tests
app.py          Gradio application and per-client submission limiting
```

## Interview-relevant design decisions

- Keep Gemini perception, Jev scoring, pipeline orchestration, ranking, and UI in separate modules with injectable provider interfaces.
- Use a strict, inspectable semantic `Context` contract as the boundary between the vision and decision models.
- Keep secrets in environment variables and use mock-only tests so normal test runs have no API cost.
- Preserve image indices and stable tie ordering, while making no-match and ambiguity explicit instead of always returning a winner.
- Treat direct-model comparisons and the no-identity ablation as evaluation questions, not as evidence already established by this implementation.
