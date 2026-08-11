"""Command-line entry point."""

import argparse
import json
from pathlib import Path

from tinymeta.config import load_thresholds
from tinymeta.profiling import CSVProfileError, profile_csv
from tinymeta.tagging import EmbeddingTagger, RuleBasedTagger, SentenceTransformerEncoder
from tinymeta.taxonomy import load_taxonomy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tinymeta")
    subparsers = parser.add_subparsers(dest="command", required=True)
    tag = subparsers.add_parser("tag", help="profile and tag a CSV")
    tag.add_argument("csv", type=Path)
    tag.add_argument("--taxonomy", type=Path, default=Path("taxonomy/environmental.yaml"))
    tag.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    tag.add_argument("--method", choices=("rules", "embeddings"), default="rules")
    tag.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        profile = profile_csv(args.csv)
        taxonomy = load_taxonomy(args.taxonomy)
        thresholds = load_thresholds(args.config)
        if args.method == "rules":
            tagger = RuleBasedTagger(thresholds)
        else:
            encoder = SentenceTransformerEncoder(args.model)
            tagger = EmbeddingTagger(encoder, thresholds, model_name=args.model)
        payload = {
            "profile": profile.model_dump(mode="json"),
            "result": tagger.tag(profile, taxonomy).model_dump(mode="json"),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    except (CSVProfileError, OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"tinymeta: {exc}") from exc


if __name__ == "__main__":
    main()
