from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path


UCI_MUSHROOM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
)

RAW_COLUMNS = [
    "label",
    "cap_shape",
    "cap_surface",
    "cap_color",
    "bruises",
    "odor",
    "gill_attachment",
    "gill_spacing",
    "gill_size",
    "gill_color",
    "stalk_shape",
    "stalk_root",
    "stalk_surface_above_ring",
    "stalk_surface_below_ring",
    "stalk_color_above_ring",
    "stalk_color_below_ring",
    "veil_type",
    "veil_color",
    "ring_number",
    "ring_type",
    "spore_print_color",
    "population",
    "habitat",
]


def read_raw_rows(raw_path: Path, url: str) -> list[list[str]]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.exists():
        with urllib.request.urlopen(url, timeout=30) as response:
            raw_path.write_bytes(response.read())

    with raw_path.open(newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    if not rows:
        raise ValueError(f"{raw_path} is empty")
    width = len(RAW_COLUMNS)
    bad = [i for i, row in enumerate(rows, 1) if len(row) != width]
    if bad:
        raise ValueError(f"unexpected column count on raw rows: {bad[:5]}")
    return rows


def one_hot_encode(rows: list[list[str]]) -> tuple[list[str], list[list[int]], dict[str, dict[str, int]]]:
    feature_values: dict[str, list[str]] = {}
    for col_idx, col_name in enumerate(RAW_COLUMNS[1:], start=1):
        feature_values[col_name] = sorted({row[col_idx] for row in rows})

    encoded_columns = ["label"]
    for col_name in RAW_COLUMNS[1:]:
        encoded_columns.extend(f"{col_name}={value}" for value in feature_values[col_name])

    category_maps = {
        col_name: {value: i for i, value in enumerate(values)}
        for col_name, values in feature_values.items()
    }
    category_maps["label"] = {"e": 0, "p": 1}

    encoded_rows: list[list[int]] = []
    for row in rows:
        encoded = [category_maps["label"][row[0]]]
        for col_idx, col_name in enumerate(RAW_COLUMNS[1:], start=1):
            value = row[col_idx]
            encoded.extend(1 if value == candidate else 0 for candidate in feature_values[col_name])
        encoded_rows.append(encoded)
    return encoded_columns, encoded_rows, category_maps


def write_encoded_csv(out_path: Path, columns: list[str], rows: list[list[int]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-out", default="data/agaricus-lepiota.data")
    parser.add_argument("--out", default="data/mushroom.csv")
    parser.add_argument("--metadata-out", default="data/mushroom_metadata.json")
    parser.add_argument("--url", default=UCI_MUSHROOM_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = read_raw_rows(Path(args.raw_out), args.url)
    columns, encoded_rows, category_maps = one_hot_encode(raw_rows)
    write_encoded_csv(Path(args.out), columns, encoded_rows)
    Path(args.metadata_out).write_text(json.dumps(category_maps, indent=2, sort_keys=True))
    print(
        f"wrote {args.out}: {len(encoded_rows)} rows, "
        f"{len(columns) - 1} one-hot features, label p=poisonous/e=edible"
    )


if __name__ == "__main__":
    main()
