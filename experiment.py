from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


@dataclass
class Node:
    feature: int = -1
    threshold: float = 0.0
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    prediction: int = 0
    is_leaf: bool = True


def deep_size(obj: Any, seen: Optional[set[int]] = None) -> int:
    """Approximate recursive Python-object memory usage in bytes."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        return size + sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return size + sum(deep_size(x, seen) for x in obj)
    if hasattr(obj, "__dict__"):
        return size + deep_size(vars(obj), seen)
    return size


def smallest_signed_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if max_value <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def gini(labels: np.ndarray, n_classes: int) -> float:
    counts = np.bincount(labels, minlength=n_classes)
    probs = counts / max(1, labels.size)
    return 1.0 - float(np.dot(probs, probs))


class DecisionTree:
    def __init__(
        self,
        max_depth: int = 8,
        min_samples_split: int = 8,
        max_features: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self.n_classes = 0
        self.root: Optional[Node] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DecisionTree":
        self.n_classes = int(np.max(y)) + 1
        self.root = self._build(x, y, depth=0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("model is not fitted")
        out = np.empty(x.shape[0], dtype=np.int16)
        for i, row in enumerate(x):
            node = self.root
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
                if node is None:
                    raise RuntimeError("corrupt tree")
            out[i] = node.prediction
        return out

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> Node:
        prediction = int(np.argmax(np.bincount(y, minlength=self.n_classes)))
        if (
            depth >= self.max_depth
            or y.size < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return Node(prediction=prediction)

        split = self._best_split(x, y)
        if split is None:
            return Node(prediction=prediction)

        feature, threshold, mask = split
        return Node(
            feature=feature,
            threshold=threshold,
            left=self._build(x[mask], y[mask], depth + 1),
            right=self._build(x[~mask], y[~mask], depth + 1),
            prediction=prediction,
            is_leaf=False,
        )

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Optional[tuple[int, float, np.ndarray]]:
        n_samples, n_features = x.shape
        feature_count = self.max_features or n_features
        features = self.rng.choice(n_features, size=feature_count, replace=False)
        parent_gini = gini(y, self.n_classes)
        best_gain = 0.0
        best_feature = -1
        best_threshold = 0.0
        best_mask: Optional[np.ndarray] = None

        for feature in features:
            values = x[:, feature]
            candidates = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
            for threshold in candidates:
                mask = values <= threshold
                left_count = int(np.sum(mask))
                right_count = n_samples - left_count
                if left_count == 0 or right_count == 0:
                    continue
                weighted = (
                    left_count * gini(y[mask], self.n_classes)
                    + right_count * gini(y[~mask], self.n_classes)
                ) / n_samples
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain = gain
                    best_feature = int(feature)
                    best_threshold = float(threshold)
                    best_mask = mask

        if best_mask is None:
            return None
        return best_feature, best_threshold, best_mask


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 8,
        min_samples_split: int = 8,
        max_features: Optional[int] = None,
        random_state: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees: list[DecisionTree] = []
        self.n_classes = 0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RandomForest":
        self.n_classes = int(np.max(y)) + 1
        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = x.shape
        max_features = self.max_features or max(1, int(math.sqrt(n_features)))
        self.trees = []
        for i in range(self.n_estimators):
            sample_idx = rng.integers(0, n_samples, size=n_samples)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=self.random_state + i + 1,
            )
            tree.fit(x[sample_idx], y[sample_idx])
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        votes = np.zeros((x.shape[0], self.n_classes), dtype=np.int16)
        for tree in self.trees:
            pred = tree.predict(x)
            votes[np.arange(x.shape[0]), pred] += 1
        return np.argmax(votes, axis=1).astype(np.int16)


@dataclass
class StaticTree:
    feature: np.ndarray
    threshold: np.ndarray
    left: np.ndarray
    right: np.ndarray
    prediction: np.ndarray
    is_leaf: np.ndarray

    @classmethod
    def from_tree(cls, tree: DecisionTree) -> "StaticTree":
        if tree.root is None:
            raise RuntimeError("model is not fitted")
        nodes: list[Node] = []
        index: dict[int, int] = {}

        def visit(node: Node) -> None:
            index[id(node)] = len(nodes)
            nodes.append(node)
            if not node.is_leaf:
                if node.left is not None:
                    visit(node.left)
                if node.right is not None:
                    visit(node.right)

        visit(tree.root)
        idx_dtype = smallest_signed_dtype(len(nodes))
        max_feature = max((n.feature for n in nodes), default=0)
        feature_dtype = smallest_signed_dtype(max(0, max_feature))
        pred_dtype = smallest_signed_dtype(tree.n_classes)

        feature = np.full(len(nodes), -1, dtype=feature_dtype)
        threshold = np.zeros(len(nodes), dtype=np.float32)
        left = np.full(len(nodes), -1, dtype=idx_dtype)
        right = np.full(len(nodes), -1, dtype=idx_dtype)
        prediction = np.zeros(len(nodes), dtype=pred_dtype)
        is_leaf = np.zeros(len(nodes), dtype=np.bool_)

        for i, node in enumerate(nodes):
            feature[i] = node.feature
            threshold[i] = node.threshold
            prediction[i] = node.prediction
            is_leaf[i] = node.is_leaf
            if node.left is not None:
                left[i] = index[id(node.left)]
            if node.right is not None:
                right[i] = index[id(node.right)]
        return cls(feature, threshold, left, right, prediction, is_leaf)

    @property
    def nbytes(self) -> int:
        return sum(
            arr.nbytes for arr in (self.feature, self.threshold, self.left, self.right, self.prediction, self.is_leaf)
        )

    @property
    def node_count(self) -> int:
        return int(self.feature.size)

    def predict(self, x: np.ndarray) -> np.ndarray:
        current = np.zeros(x.shape[0], dtype=self.left.dtype)
        active = ~self.is_leaf[current]
        while np.any(active):
            rows = np.flatnonzero(active)
            nodes = current[rows]
            go_left = x[rows, self.feature[nodes]] <= self.threshold[nodes]
            current[rows] = np.where(go_left, self.left[nodes], self.right[nodes])
            active[rows] = ~self.is_leaf[current[rows]]
        return self.prediction[current]


class StaticForest:
    def __init__(self, trees: list[StaticTree], n_classes: int) -> None:
        self.trees = trees
        self.n_classes = n_classes

    @classmethod
    def from_forest(cls, forest: RandomForest) -> "StaticForest":
        return cls([StaticTree.from_tree(tree) for tree in forest.trees], forest.n_classes)

    @property
    def nbytes(self) -> int:
        return sum(tree.nbytes for tree in self.trees)

    @property
    def node_count(self) -> int:
        return sum(tree.node_count for tree in self.trees)

    def predict(self, x: np.ndarray) -> np.ndarray:
        votes = np.zeros((x.shape[0], self.n_classes), dtype=np.int16)
        for tree in self.trees:
            pred = tree.predict(x)
            votes[np.arange(x.shape[0]), pred] += 1
        return np.argmax(votes, axis=1).astype(np.int16)


def make_synthetic(samples: int, features: int, classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(samples, features)).astype(np.float32)
    weights = rng.normal(size=(features, classes))
    logits = x @ weights + 0.35 * rng.normal(size=(samples, classes))
    nonlinear = np.sin(x[:, : min(4, features)]).sum(axis=1)
    logits[:, 0] += nonlinear
    y = np.argmax(logits, axis=1).astype(np.int16)
    return x, y


def load_csv(path: str, label_column: str) -> tuple[np.ndarray, np.ndarray]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    feature_names = [name for name in reader.fieldnames or [] if name != label_column]
    labels_raw = [row[label_column] for row in rows]
    label_map = {value: i for i, value in enumerate(sorted(set(labels_raw)))}
    y = np.array([label_map[v] for v in labels_raw], dtype=np.int16)
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    return x, y


def train_test_split(
    x: np.ndarray, y: np.ndarray, test_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(x.shape[0])
    test_size = max(1, int(x.shape[0] * test_ratio))
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


def time_call(fn: Any, repeats: int = 5) -> tuple[Any, float]:
    result = fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, min(times)


def format_bytes(n: float) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def run(args: argparse.Namespace) -> None:
    if args.csv:
        dataset_name = args.csv
        x, y = load_csv(args.csv, args.label)
    else:
        default_dataset = "data/mushroom.csv"
        dataset_name = default_dataset
        try:
            x, y = load_csv(default_dataset, args.label)
            print(f"loaded default dataset: {default_dataset}")
            print()
        except FileNotFoundError:
            dataset_name = "synthetic fallback"
            x, y = make_synthetic(args.samples, args.features, args.classes, args.seed)
    x_train, x_test, y_train, y_test = train_test_split(x, y, args.test_ratio, args.seed)

    if args.model == "tree":
        model: Any = DecisionTree(args.max_depth, args.min_samples_split, None, args.seed).fit(x_train, y_train)
        static_model: Any = StaticTree.from_tree(model)
    else:
        model = RandomForest(
            n_estimators=args.trees,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=args.seed,
        ).fit(x_train, y_train)
        static_model = StaticForest.from_forest(model)

    obj_pred, obj_time = time_call(lambda: model.predict(x_test), args.repeats)
    static_pred, static_time = time_call(lambda: static_model.predict(x_test), args.repeats)

    if not np.array_equal(obj_pred, static_pred):
        raise AssertionError("static representation changed predictions")

    accuracy = float(np.mean(static_pred == y_test))
    object_bytes = deep_size(model)
    static_bytes = static_model.nbytes
    memory_saved = 1.0 - static_bytes / object_bytes
    time_saved = 1.0 - static_time / obj_time

    metrics = {
        "dataset": dataset_name,
        "model": args.model,
        "train_rows": x_train.shape[0],
        "test_rows": x_test.shape[0],
        "features": x.shape[1],
        "classes": int(np.max(y)) + 1,
        "nodes": static_model.node_count,
        "accuracy": accuracy,
        "object_memory_bytes": object_bytes,
        "static_memory_bytes": static_bytes,
        "memory_saved_percent": memory_saved * 100,
        "object_predict_ms": obj_time * 1000,
        "static_predict_ms": static_time * 1000,
        "time_saved_percent": time_saved * 100,
    }

    print("Experiment result")
    print("=================")
    print(f"dataset:            {dataset_name}")
    print(f"model:              {args.model}")
    print(f"train/test rows:    {x_train.shape[0]} / {x_test.shape[0]}")
    print(f"features/classes:   {x.shape[1]} / {int(np.max(y)) + 1}")
    print(f"nodes:              {static_model.node_count}")
    print(f"accuracy:           {accuracy:.4f}")
    print()
    print(f"object memory:      {format_bytes(object_bytes)}")
    print(f"static memory:      {format_bytes(static_bytes)}")
    print(f"memory saved:       {memory_saved * 100:.2f}%")
    print()
    print(f"object predict:     {obj_time * 1000:.3f} ms")
    print(f"static predict:     {static_time * 1000:.3f} ms")
    print(f"time saved:         {time_saved * 100:.2f}%")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            writer.writeheader()
            writer.writerow(metrics)
        print()
        print(f"wrote report:       {report_path}")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["tree", "forest"], default="forest")
    parser.add_argument("--csv", help="Optional numeric CSV dataset path. Defaults to data/mushroom.csv when present.")
    parser.add_argument("--label", default="label", help="CSV label column name.")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--trees", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=8)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--report", help="Optional CSV path for saving one-row experiment metrics.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
