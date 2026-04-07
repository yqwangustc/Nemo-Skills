"""Combine per-seed output-rs*.jsonl files into a single jsonl with aggregated predictions."""

import json
import glob
from collections import defaultdict

input_dir = "/lustre/fsw/portfolios/llmservice/users/yongqiangw/data/music_qa/reasoning_filter_infer.t2"
output_file = f"{input_dir}/combined.jsonl"

# Open all seed files
seed_files = [f"{input_dir}/output-rs{i}.jsonl" for i in range(5)]
print(f"Using {len(seed_files)} seed files")

# Stream line-by-line across all files simultaneously
handles = [open(fpath) for fpath in seed_files]
count = 0
with open(output_file, "w") as out:
    for lines in zip(*handles):
        rows = [json.loads(line) for line in lines]
        # Verify all seeds have the same question at this index
        questions = {r["question"] for r in rows}
        assert len(questions) == 1, f"Row {count}: question mismatch across seeds"
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
        count += 1
        if count % 500000 == 0:
            print(f"  Processed {count} rows...")

for h in handles:
    h.close()

print(f"Wrote {count} rows to {output_file}")
