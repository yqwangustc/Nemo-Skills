# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import re

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def _extract_letter(text):
    """Extract the answer letter from a multiple-choice style string.

    Handles formats like "(A) xxx", "A) xxx", "(a)", "A", etc.
    """
    if not text:
        return None

    m = re.match(r"\s*\(?\s*([A-Za-z])\s*\)\s*", text)
    return m.group(1).upper() if m else None


def eval_simple_mcq(cfg):
    eval_config = BaseEvaluatorConfig(**cfg)

    jsonl_file = eval_config.input_file
    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in tqdm(data, desc="Evaluating simple_mcq"):
            generation = (sample.get("generation") or "").strip()
            predicted = _extract_letter(generation)
            sample["predicted_answer"] = predicted

            # Support both 'expected_answer' and 'answer' fields
            expected_raw = sample.get("expected_answer") or sample.get("answer", "")
            expected = _extract_letter(str(expected_raw).strip())

            if predicted is not None and expected is not None:
                sample["symbolic_correct"] = predicted == expected
            else:
                sample["symbolic_correct"] = False

            # Ensure expected_answer is set for downstream metrics
            if "expected_answer" not in sample and expected is not None:
                sample["expected_answer"] = expected

            LOG.info(
                f"predicted_answer={predicted}, expected_answer={expected}, "
                f"symbolic_correct={sample['symbolic_correct']}"
            )

            fout.write(json.dumps(sample) + "\n")
