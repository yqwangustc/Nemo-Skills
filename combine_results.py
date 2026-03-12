"""Combine per-seed output-rs*.jsonl files into a single jsonl with aggregated predictions."""

import json
import glob
from collections import defaultdict

input_dir = "/home/yongqiangw/work/tmp/output_v2"
output_file = "/home/yongqiangw/work/tmp/combined.jsonl"

# Read all seed files
seed_files = sorted(glob.glob(f"{input_dir}/output-rs*.jsonl"))
print(f"Found {len(seed_files)} seed files")

# Group rows by question (using row index as key since order is consistent)
rows_by_idx = defaultdict(list)
for fpath in seed_files:
    with open(fpath) as f:
        for idx, line in enumerate(f):
            rows_by_idx[idx].append(json.loads(line))

# Combine
with open(output_file, "w") as out:
    for idx in sorted(rows_by_idx.keys()):
        rows = rows_by_idx[idx]
        # Verify all seeds have the same question at this index
        questions = {r["question"] for r in rows}
        assert len(questions) == 1, f"Row {idx}: question mismatch across seeds"
        first = rows[0]
        predictions = [r["_full_generation"] for r in rows]
        correct_count = sum(1 for r in rows if r.get("symbolic_correct"))
        combined = {
            "caption": first["caption"],
            "sound": first["sound"],
            "question": first["question"],
            "answer": first["answer"],
            "question_type": first["question_type"],
            "dataset": first["dataset"],
            "nemotron_predictions": predictions,
            "pass_rate": correct_count / len(rows),
        }
        out.write(json.dumps(combined) + "\n")

print(f"Wrote {len(rows_by_idx)} rows to {output_file}")
