# Milestone 1 design review

## Simplest technically sound architecture

TinyMeta treats raw data inspection and semantic inference as separate stages:

```text
CSV -> deterministic profile -> tagger strategy -> validated tags + provenance
                 |                    |
                 |              rules or embeddings
                 +---- taxonomy + thresholds ----+
```

The profile is a bounded, serializable contract. Taggers share one interface and only emit
terms that came from a validated taxonomy. Confidence thresholds are policy, not model code.
This keeps the first implementation testable on a laptop and leaves zero-shot/fine-tuning,
feedback storage, search, and benchmarking outside the milestone until there is evidence they
are needed.

## Milestone 1 acceptance boundary

The first end-to-end milestone accepts a local CSV, deterministically extracts schema and basic
statistics, loads a versioned environmental taxonomy, applies a transparent rules baseline,
assigns review status from configurable thresholds, and returns the profile plus provenance as
validated JSON through `tinymeta tag`. The minimal ML vertical slice substitutes embedding
similarity behind the same interface and lazily loads its optional local model.

Not in milestone 1: zero-shot inference, training, a UI, persistence, search, benchmark runners,
quantization, or fabricated performance claims.

## Assumptions and risks

- CSVs fit in local memory. Pandas is the simplest dependable implementation; chunked profiling
  is future work for large files.
- Pandas dtype inference is useful but not semantic truth. Identifier-like numbers and locale-
  formatted values may be misclassified.
- Column names and filenames carry signal. Opaque, abbreviated, multilingual, or misleading
  schemas weaken both rules and embeddings.
- Cosine similarity is a ranking score, not a calibrated probability. Thresholds are provisional
  policy defaults and must be tuned against held-out data before auto-accept is trusted.
- A closed taxonomy can force a nearest-but-wrong concept. TinyMeta filters low scores and permits
  an empty result, but true out-of-distribution detection remains experimental work.
- Sample values can expose sensitive data. The profiler bounds the sample but does not redact it;
  deployments need a data-handling policy.
- Taxonomy aliases are data-driven, but the rules baseline remains lexical and can produce valid
  multi-label ambiguity (for example traffic plus NO2).
- Synthetic records demonstrate the contract, not external validity. Real, independently labelled
  test data is required for research claims.

## Embedding candidates reviewed (11 August 2026)

These are experiment candidates, not claimed winners. Details and licenses were checked against
their Hugging Face model cards; downstream users should re-check them before redistribution.

| Model | Scope / size signal | License | Trade-off for TinyMeta |
|---|---|---|---|
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | English, 384 dimensions, 256-token input, roughly 90 MB weights | Apache-2.0 | Mature, small default for the slice; shorter context and older general-purpose training. |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) | English, 384 dimensions, 512-token input, 33.4M parameters | MIT | Strong compact retrieval baseline; task framing/prefix choices must be held constant in evaluation. |
| [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) | English, 384 dimensions, 512-token input, 33.4M parameters | MIT | Very small reported footprint (~0.07 GB); English-only and long profiles are truncated. |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 94 languages, 384 dimensions, 12 layers, ~471 MB weights | MIT | Best candidate here for multilingual schemas; materially larger and expects E5-style input prefixes. |

Model-card benchmark scores are not TinyMeta results. Domain fit, latency, peak memory,
calibration, and classification quality must be measured on the same locked evaluation split.

## Evaluation representation

`data/evaluation/examples.jsonl` is versioned JSON Lines: one independently addressable example
per row with stable ID, split, input reference or compact profile, controlled multi-label ground
truth, and annotator notes. Empty labels represent out-of-taxonomy cases. Future datasets should
add taxonomy version and annotation provenance; split assignment must group related sources to
prevent leakage.
