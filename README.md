# TinyMeta

TinyMeta is a local-first research prototype for assigning controlled metadata tags to scientific
CSV datasets. It profiles a dataset before classification, never sends data to an external LLM
API, and attaches evidence, taxonomy paths, method, confidence, and review status to every tag.

## The problem and question

Scientific data often arrives with only a filename and cryptic schema. Manual cataloguing is slow,
while unconstrained text generation is hard to govern. TinyMeta asks: **how effectively can small,
local embedding models classify scientific datasets into a controlled taxonomy, and when should
the system abstain?**

## What exists now

Milestone 1 is intentionally compact:

- deterministic CSV profiling (shape, types, missingness, cardinality, numeric statistics, bounded
  samples, likely timestamps/geography, and units encoded in names);
- validated YAML taxonomy and configurable review thresholds;
- a traceable lexical baseline;
- an embedding-similarity tagger behind the same interface, with lazy optional model loading;
- validated Pydantic contracts and a JSON CLI;
- realistic synthetic/evaluation examples and download-free unit tests.

See [the design review](docs/design.md) for architecture, assumptions, risks, model candidates,
evaluation representation, and the exact milestone boundary.

## Architecture

```text
                     local CSV
                         |
                 deterministic profiler
                         |
                  DatasetProfile
                         |
            +------------+------------+
            |                         |
       lexical rules             local encoder
            |                  + cosine ranking
            +------------+------------+
                         |
              controlled taxonomy only
                         |
        confidence policy + provenance + JSON
```

Models receive the compact profile, not the full table. The taxonomy and confidence policy are
configuration, and taggers implement one interface.

## Quick start

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
tinymeta tag data/synthetic/amsterdam_air_quality.csv
```

Embedding mode is optional and downloads the selected model on first use:

```bash
python -m pip install -e '.[embeddings]'
tinymeta tag data/synthetic/amsterdam_air_quality.csv --method embeddings
```

Use `--taxonomy`, `--config`, or `--model` to replace defaults. No model is downloaded for the
default rules command or unit tests.

## Example

The included station CSV produces a profile and controlled tags such as
`nitrogen_dioxide`, `particulate_matter_pm25`, `temperature`, and `humidity`. A tag resembles:

```json
{
  "tag": "nitrogen_dioxide",
  "confidence": 0.98,
  "method": "rules",
  "evidence": ["column:no2_ug_m3"],
  "taxonomy_path": ["environment", "atmospheric", "air_quality", "pollutants", "nitrogen_dioxide"],
  "status": "auto_accept"
}
```

## Experiments and results

Planned experiments compare rules and candidate encoders under fixed context ablations, then add
zero-shot and fine-tuned baselines only when evaluation data is adequate. Metrics will include
multi-label precision/recall/F1, exact match, coverage/abstention, threshold calibration, latency,
peak memory, and model size.

| Method | Precision | Recall | F1 | Exact match | Latency | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|
| Rules | TBD | TBD | TBD | TBD | TBD | TBD |
| all-MiniLM-L6-v2 embeddings | TBD | TBD | TBD | TBD | TBD | TBD |

`TBD` is deliberate: no benchmark has run, so there are no performance claims yet.

## Failure cases and responsible use

Expected failures include opaque abbreviations, multilingual schemas, misleading names, unseen
scientific variables, and ambiguous datasets. Similarity is not probability. The default thresholds
are uncalibrated, and even `auto_accept` output requires validation before production use. Unknown
concepts may correctly yield no tags. Samples may contain sensitive values; do not expose CLI output
without reviewing the data-handling context.

## Design decisions and current limitations

Pandas minimizes implementation complexity and makes profiling behavior familiar. Pydantic makes
every boundary explicit. JSON output enables composition without prematurely adding a web service or
database. Model dependencies are optional so deterministic workflows stay lightweight.

Current profiling loads a whole CSV into memory. Rules only use names, embedding evidence is profile-
level rather than feature attribution, evaluation data is tiny and synthetic, confidence is not
calibrated, and no zero-shot/fine-tuning/search/feedback workflow is implemented yet.

## Development

```bash
ruff format --check .
ruff check .
pytest -m 'not integration'
```

An opt-in integration test can be run after installing the embedding extra:

```bash
pytest -m integration
```

## What we learned

The first milestone establishes that useful provenance and abstention contracts do not require a
generative model. Whether embeddings improve on transparent aliases remains an empirical question.

## Future work

Build a leakage-controlled labelled corpus; run context ablations and model comparisons; calibrate
thresholds for a target precision; add out-of-distribution tests, multilingual schemas, streaming
profiles, correction storage, and then—only if justified—fine-tuning and ONNX/quantized CPU inference.
