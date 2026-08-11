# Evaluation data contract

Each UTF-8 JSONL record is a `tinymeta-evaluation/v1` example with a stable `id`, a
`split` (`train`, `validation`, or `test`), a relative CSV path, controlled `labels`, and
`notes`. Splits are assigned by dataset family/source—not by rows—to avoid near-duplicate
leakage. Labels must exist in the referenced taxonomy. Ambiguous multi-label and deliberate
unknown/adversarial examples are supported; `labels: []` means no in-taxonomy label applies.

The starter file is illustrative evaluation data, not a benchmark large enough for claims.
