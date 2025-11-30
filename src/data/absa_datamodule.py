from typing import Any, Dict, Optional, Tuple
import pickle
from pathlib import Path
from collections import Counter

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from src.data.components.absa_dataset import ABSADataset, collate_fn


class Vocabulary:
    """Vocabulary class for mapping between tokens and indices."""

    def __init__(self, counter: Counter, special_tokens: list = None):
        if special_tokens is None:
            special_tokens = ["<pad>", "<unk>"]

        self.pad_index = 0
        self.unk_index = 1

        # Create index-to-string mapping starting with special tokens
        self.itos = special_tokens.copy()

        # Remove special tokens from counter if present
        counter = counter.copy()
        for token in special_tokens:
            counter.pop(token, None)

        # Sort by frequency (descending) then alphabetically
        sorted_items = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        # Add words to vocabulary
        for word, _ in sorted_items:
            self.itos.append(word)

        # Create string-to-index mapping
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        """Get index for token, return unk_index if not found."""
        return self.stoi.get(token, self.unk_index)

    def extend(self, other_vocab: "Vocabulary") -> "Vocabulary":
        """Extend this vocabulary with another vocabulary."""
        for token in other_vocab.itos:
            if token not in self.stoi:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1
        return self

    def save(self, file_path: Path) -> None:
        """Save vocabulary to file."""
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path: Path) -> "Vocabulary":
        """Load vocabulary from file."""
        with open(file_path, "rb") as f:
            return pickle.load(f)


class ABSADataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        vocab_dir: str = "data/vocab/",
        bert_model_name: str = "bert-base-uncased",
        max_len: int = 128,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.vocab_dict: Optional[Dict] = None
        self.tokenizer: Optional[BertTokenizer] = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes."""
        return 3  # positive, negative, neutral

    @property
    def dep_vocab_size(self) -> int:
        """Get the size of dependency relation vocabulary."""
        if self.vocab_dict is None:
            self.vocab_dict = self._load_vocabularies()
        return len(self.vocab_dict["dep"])

    @property
    def pos_vocab_size(self) -> int:
        """Get the size of POS tag vocabulary."""
        if self.vocab_dict is None:
            self.vocab_dict = self._load_vocabularies()
        return len(self.vocab_dict["pos"])

    @property
    def token_vocab_size(self) -> int:
        """Get the size of token vocabulary."""
        if self.vocab_dict is None:
            self.vocab_dict = self._load_vocabularies()
        return len(self.vocab_dict["token"])

    def prepare_data(self) -> None:
        """Download data if needed."""
        BertTokenizer.from_pretrained(self.hparams.bert_model_name)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`."""
        # Load vocabulary first
        if not self.vocab_dict:
            self.vocab_dict = self._load_vocabularies()

        # Load tokenizer
        if not self.tokenizer:
            self.tokenizer = BertTokenizer.from_pretrained(self.hparams.bert_model_name)

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if not self.data_train:
                self.data_train = ABSADataset(
                    data_path=Path(self.hparams.data_dir) / "train.json",
                    vocab_dict=self.vocab_dict,
                    bert_model_name=self.hparams.bert_model_name,
                    max_len=self.hparams.max_len,
                )

            if not self.data_val:
                self.data_val = ABSADataset(
                    data_path=Path(self.hparams.data_dir) / "test.json",
                    vocab_dict=self.vocab_dict,
                    bert_model_name=self.hparams.bert_model_name,
                    max_len=self.hparams.max_len,
                )

        # Assign test datasets for use in dataloaders
        if stage == "test" or stage is None:
            if not self.data_test:
                self.data_test = ABSADataset(
                    data_path=Path(self.hparams.data_dir) / "test.json",
                    vocab_dict=self.vocab_dict,
                    bert_model_name=self.hparams.bert_model_name,
                    max_len=self.hparams.max_len,
                )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up after fit or test."""
        pass

    def state_dict(self) -> Dict[str, Any]:
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Things to do when loading checkpoint."""
        pass

    def _load_vocabularies(self) -> Dict[str, Any]:
        """Load all vocabularies from files."""
        vocab_dir = Path(self.hparams.vocab_dir)

        vocab_files = {
            "token": "vocab_tok.vocab",
            "pos": "vocab_pos.vocab",
            "dep": "vocab_dep.vocab",
            "position": "vocab_post.vocab",
            "polarity": "vocab_pol.vocab",
        }

        vocab_dict = {}
        for vocab_type, filename in vocab_files.items():
            vocab_path = vocab_dir / filename
            if vocab_path.exists():
                with open(vocab_path, "rb") as f:
                    vocab_dict[vocab_type] = pickle.load(f)
                print(
                    f"Loaded {vocab_type} vocabulary with size: {len(vocab_dict[vocab_type])}"
                )
            else:
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        return vocab_dict
