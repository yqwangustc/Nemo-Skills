"""End-to-end pass@k evaluation pipeline using NeMo Skills.

Generates k samples per problem, judges correctness with simple_mcq evaluator,
and computes pass@k / majority@k metrics.

Usage (Python API):
    python eval_pipeline.py

Modify the configuration variables below to match your setup.
"""

from nemo_skills.pipeline.cli import generate, summarize_results, wrap_arguments

# ---------------------------------------------------------------------------
# Configuration — edit these to match your setup
# ---------------------------------------------------------------------------
cluster = "iad"
model = "/path/to/your/model"
input_file = "/path/to/your/input.jsonl"
output_dir = "/path/to/your/output_dir"
k = 8  # number of samples per problem (controls pass@k)

server_type = "vllm"
server_gpus = 1
server_nodes = 1
num_chunks = 128

# Inference parameters
temperature = 1.0
top_p = 0.95
tokens_to_generate = 2048

# Additional server args (adjust for your model)
server_args = (
    "--max-num-seqs=256 "
    "--swap-space=8 "
    "--gpu-memory-utilization=0.9 "
    "--max-model-len=65536"
)

expname = "pass_at_k_eval"

# ---------------------------------------------------------------------------
# Step 1: Generate k samples with simple_mcq evaluation
# ---------------------------------------------------------------------------
# num_random_seeds=k creates output-rs0.jsonl ... output-rs{k-1}.jsonl
# eval_type=simple_mcq runs the evaluator after each generation, adding
# predicted_answer and symbolic_correct fields.

generate(
    cluster=cluster,
    server_type=server_type,
    model=model,
    server_args=server_args,
    server_gpus=server_gpus,
    server_nodes=server_nodes,
    num_chunks=num_chunks,
    num_random_seeds=k,
    input_file=input_file,
    output_dir=output_dir,
    expname=expname,
    ctx=wrap_arguments(
        "++eval_type=simple_mcq "
        "++prompt_format=openai "
        f"++inference.temperature={temperature} "
        f"++inference.top_p={top_p} "
        f"++inference.tokens_to_generate={tokens_to_generate} "
        "++parse_reasoning=False "
    ),
)

# ---------------------------------------------------------------------------
# Step 2: Compute pass@k metrics from the generated output-rs*.jsonl files
# ---------------------------------------------------------------------------
# metric_type="multichoice" uses MathMetrics which computes pass@1..k and
# majority@1..k via BaseMetrics._compute_pass_at_k().

summarize_results(
    results_dir=output_dir,
    metric_type="multichoice",
)
