"""
Prepare vocabulary and initial word vectors for relation extraction.

Author: jiuhan chen
Date: 2025年9月30日
"""

import json
import pickle
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Import Vocabulary from the datamodule where it's defined for pickle compatibility
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.absa_datamodule import Vocabulary


class DataProcessor:
    """Process data and build vocabularies."""

    def __init__(self, data_dir: Path, lowercase: bool = True):
        self.data_dir = Path(data_dir)
        self.lowercase = lowercase

    def load_data_files(self) -> Tuple[List[Dict], List[Dict]]:
        """Load train and test data files."""
        train_file = self.data_dir / "train.json"
        test_file = self.data_dir / "test.json"

        with open(train_file, "r", encoding="utf-8") as f:
            train_data = json.load(f)
        with open(test_file, "r", encoding="utf-8") as f:
            test_data = json.load(f)

        return train_data, test_data

    def extract_features(self, data: List[Dict]) -> Dict[str, List]:
        """Extract features from data."""
        features = {"tokens": [], "pos_tags": [], "dep_rels": [], "max_length": 0}

        for item in data:
            tokens = item["token"]
            if self.lowercase:
                tokens = [t.lower() for t in tokens]

            features["tokens"].extend(tokens)
            features["pos_tags"].extend(item["pos"])
            features["dep_rels"].extend(item["deprel"])
            features["max_length"] = max(features["max_length"], len(tokens))

        return features

    def build_vocabularies(
        self, train_features: Dict, test_features: Dict
    ) -> Dict[str, Vocabulary]:
        """Build all vocabularies from features."""
        all_tokens = train_features["tokens"] + test_features["tokens"]
        all_pos = train_features["pos_tags"] + test_features["pos_tags"]
        all_dep = train_features["dep_rels"] + test_features["dep_rels"]
        max_len = max(train_features["max_length"], test_features["max_length"])

        token_counter = Counter(all_tokens)
        pos_counter = Counter(all_pos)  # 保留真正的词性统计
        dep_counter = Counter(all_dep)
        # 位置索引（包含 -max_len 到 +max_len），如不想包含最外层可改回 range(-max_len, max_len)
        position_counter = Counter(range(-max_len, max_len + 1))
        pol_counter = Counter(["positive", "negative", "neutral"])

        vocabs = {
            "token": Vocabulary(token_counter),
            "pos": Vocabulary(pos_counter),
            "dep": Vocabulary(dep_counter),
            "position": Vocabulary(position_counter),
            "polarity": Vocabulary(pol_counter, special_tokens=[]),
        }
        return vocabs

    def process(self) -> Dict[str, Vocabulary]:
        """Main processing method."""
        print("Loading data files...")
        train_data, test_data = self.load_data_files()

        print("Extracting features...")
        train_features = self.extract_features(train_data)
        test_features = self.extract_features(test_data)

        print(f"Train examples: {len(train_data)}")
        print(f"Test examples: {len(test_data)}")
        print(
            f"Max sequence length: {max(train_features['max_length'], test_features['max_length'])}"
        )

        print("Building vocabularies...")
        vocabs = self.build_vocabularies(train_features, test_features)

        return vocabs


def save_vocabularies(vocabs: Dict[str, Vocabulary], output_dir: Path) -> None:
    """Save all vocabularies to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_files = {
        "token": "vocab_tok.vocab",
        "pos": "vocab_pos.vocab",
        "dep": "vocab_dep.vocab",
        "position": "vocab_post.vocab",
        "polarity": "vocab_pol.vocab",
    }

    for vocab_type, vocab in vocabs.items():
        filename = vocab_files[vocab_type]
        file_path = output_dir / filename
        vocab.save(file_path)
        print(f"Saved {vocab_type} vocabulary ({len(vocab)} items) to {file_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare vocab for relation extraction."
    )
    parser.add_argument(
        "--data_dir", type=Path, required=True, help="TACRED data directory."
    )
    parser.add_argument(
        "--vocab_dir", type=Path, required=True, help="Output vocab directory."
    )
    parser.add_argument(
        "--lower",
        action="store_true",
        default=True,
        help="Lowercase all words (default: True).",
    )
    parser.add_argument(
        "--no_lower", action="store_false", dest="lower", help="Do not lowercase words."
    )

    args = parser.parse_args()

    # Process data and build vocabularies
    processor = DataProcessor(args.data_dir, lowercase=args.lower)
    vocabs = processor.process()

    # Save vocabularies
    print("\nSaving vocabularies...")
    save_vocabularies(vocabs, args.vocab_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()
