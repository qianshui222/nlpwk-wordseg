from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression


def words_to_boundaries(words: Sequence[str]) -> List[int]:
    text = "".join(words)
    boundaries = [0] * max(0, len(text) - 1)
    position = 0
    for word in words[:-1]:
        position += len(word)
        boundaries[position - 1] = 1
    return boundaries


def boundaries_to_words(text: str, boundaries: Sequence[int]) -> List[str]:
    if not text:
        return []
    words: List[str] = []
    start = 0
    for index, label in enumerate(boundaries):
        if label == 1:
            words.append(text[start : index + 1])
            start = index + 1
    words.append(text[start:])
    return words


def load_segmented_corpus(path: Path) -> List[List[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip().split() for line in lines if line.strip()]


def boundary_set(words: Sequence[str]) -> set[int]:
    offsets = set()
    position = 0
    for word in words[:-1]:
        position += len(word)
        offsets.add(position - 1)
    return offsets


@dataclass
class BoundaryClassifier:
    dictionary: set[str]
    max_len: int
    training_words: set[str] | None = None
    vectorizer: DictVectorizer | None = None
    model: LogisticRegression | None = None

    def train(self, segmented_sentences: Iterable[Sequence[str]], segmenter) -> None:
        features: List[Dict[str, object]] = []
        labels: List[int] = []
        self.training_words = set()

        for words in segmented_sentences:
            self.training_words.update(words)
            text = "".join(words)
            if len(text) < 2:
                continue
            boundaries = words_to_boundaries(words)
            for index, label in enumerate(boundaries):
                features.append(self._extract_features(text, index, segmenter))
                labels.append(label)

        self.vectorizer = DictVectorizer(sparse=True)
        matrix = self.vectorizer.fit_transform(features)
        self.model = LogisticRegression(max_iter=1000, solver="liblinear")
        self.model.fit(matrix, labels)

    def predict(self, text: str, segmenter) -> List[str]:
        if self.vectorizer is None or self.model is None or len(text) < 2:
            return [text] if text else []

        feature_list = [self._extract_features(text, index, segmenter) for index in range(len(text) - 1)]
        matrix = self.vectorizer.transform(feature_list)
        predictions = self.model.predict(matrix).tolist()
        probabilities = self.model.predict_proba(matrix)[:, 1].tolist()
        words = boundaries_to_words(text, predictions)
        return self._merge_with_training_words(words, probabilities)

    def save(self, path: Path) -> None:
        if self.vectorizer is None or self.model is None:
            raise ValueError("model has not been trained")
        payload = {
            "dictionary": sorted(self.dictionary),
            "max_len": self.max_len,
            "training_words": sorted(self.training_words or set()),
            "vectorizer": self.vectorizer,
            "model": self.model,
        }
        with path.open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: Path) -> "BoundaryClassifier":
        with path.open("rb") as file:
            payload = pickle.load(file)
        return cls(
            dictionary=set(payload["dictionary"]),
            max_len=payload["max_len"],
            training_words=set(payload.get("training_words", [])),
            vectorizer=payload["vectorizer"],
            model=payload["model"],
        )

    def _extract_features(self, text: str, index: int, segmenter) -> Dict[str, object]:
        left_char = text[index]
        right_char = text[index + 1]
        prev_char = text[index - 1] if index > 0 else "<BOS>"
        next_char = text[index + 2] if index + 2 < len(text) else "<EOS>"

        forward = segmenter.forward_max_match(text)
        backward = segmenter.backward_max_match(text)
        bidirectional = segmenter.bidirectional_max_match(text)

        forward_boundaries = boundary_set(forward)
        backward_boundaries = boundary_set(backward)
        bidirectional_boundaries = boundary_set(bidirectional)

        features: Dict[str, object] = {
            "left_char": left_char,
            "right_char": right_char,
            "prev_char": prev_char,
            "next_char": next_char,
            "char_bigram": left_char + right_char,
            "left_bigram": prev_char + left_char,
            "right_bigram": right_char + next_char,
            "char_trigram": prev_char + left_char + right_char,
            "right_trigram": left_char + right_char + next_char,
            "fmm_cut": index in forward_boundaries,
            "bmm_cut": index in backward_boundaries,
            "bm_cut": index in bidirectional_boundaries,
            "left_in_dict": left_char in self.dictionary,
            "right_in_dict": right_char in self.dictionary,
            "pair_in_dict": (left_char + right_char) in self.dictionary,
            "pair_in_training": (left_char + right_char) in (self.training_words or set()),
        }

        for length in range(2, min(self.max_len, 4) + 1):
            start = max(0, index - length + 1)
            end = min(len(text), index + length + 1)
            left_span = text[start : index + 1]
            right_span = text[index + 1 : end]
            features[f"left_span_{length}_in_dict"] = left_span in self.dictionary
            features[f"right_span_{length}_in_dict"] = right_span in self.dictionary
            features[f"left_span_{length}_in_training"] = left_span in (self.training_words or set())
            features[f"right_span_{length}_in_training"] = right_span in (self.training_words or set())

        return features

    def _merge_with_training_words(self, words: List[str], probabilities: Sequence[float]) -> List[str]:
        if not self.training_words:
            return words

        merged: List[str] = []
        index = 0
        prob_index = 0
        while index < len(words):
            if len(words[index]) != 1:
                merged.append(words[index])
                prob_index += len(words[index])
                index += 1
                continue

            best_word = words[index]
            best_length = 1
            chunk = words[index]
            scan = index + 1
            local_probs: List[float] = []
            while scan < len(words) and len(words[scan]) == 1 and len(chunk) < 4:
                local_probs.append(probabilities[prob_index + len(chunk) - 1])
                chunk += words[scan]
                if chunk in self.training_words and all(prob < 0.6 for prob in local_probs):
                    best_word = chunk
                    best_length = scan - index + 1
                scan += 1

            merged.append(best_word)
            prob_index += len(best_word)
            index += best_length

        return merged
