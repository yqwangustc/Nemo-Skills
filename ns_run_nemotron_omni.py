#!/lustre/fsw/portfolios/convai/users/yifanp/miniforge3/envs/sdg_omni/bin/python

from pathlib import Path

from nemo_skills.pipeline.cli import wrap_arguments, generate


input_file = "/lustre/fsw/portfolios/convai/users/yifanp/projects/SDG_omni/sdg_data/openai_format/sound_music_qa_combined.jsonl"
output_dir = "/lustre/fsw/portfolios/convai/users/yifanp/projects/SDG_omni/sdg_data/openai_format/nemotron_omni_pred"
model = "/workspace/work/tmp/models/SDG_omni/matthieu_sft_omni/iter_0040000/mcore_to_hf"
num_chunks = 128
reasoning = False


if reasoning:
    inference_params = "++inference.temperature=0.6 ++inference.top_p=0.95 ++inference.tokens_to_generate=16384 ++inference.extra_body.skip_special_tokens=False ++parse_reasoning=True ++system_message='/think' "
else:
    inference_params = "++inference.temperature=1.0 ++inference.top_k=1 ++inference.tokens_to_generate=2048 ++parse_reasoning=False ++system_message='/no_think' "

# generate_postprocess_cmd = (
#     "/lustre/fsw/portfolios/llmservice/users/yifanp/miniforge3/envs/ns_omni_moe/bin/python "
#     "/lustre/fsw/portfolios/llmservice/users/yifanp/projects/mcore_omni_moe/score_voicebench.py "
#     f"--eval_file {output_dir / 'output.jsonl'} "
#     f"--dataset_name {dataset} "
#     f"--nproc 4 "
# )
generate_expname = "run_nemotron_omni"

generate(
    cluster="iad",
    server_type="vllm",
    model=model,
    # NOTE: nemo-skills' `serve_vllm.py` re-joins unknown args into a shell command.
    # JSON arguments must therefore:
    # - contain no spaces (otherwise they get split),
    # - escape double quotes (otherwise the shell strips them and JSON becomes invalid).
    server_args=r"""--max-num-seqs=256 --swap-space=8 --gpu-memory-utilization=0.9 --allowed-local-media-path=/ '--limit-mm-per-prompt={\"video\":0,\"image\":0,\"audio\":4}' --max-model-len=65536""",
    server_gpus=1,
    server_nodes=1,
    num_chunks=num_chunks,
    input_file=str(input_file),
    output_dir=str(output_dir),
    ctx=wrap_arguments(
        "++server.enable_soft_fail=True "
        "++prompt_format=openai "
        "++drop_content_types=[] "
        "++inference.timeout=600 "
        "++max_concurrent_requests=512 "
        + inference_params
    ),
    # postprocess_cmd=generate_postprocess_cmd,
    expname=generate_expname,
    # dependent_jobs=4,
)
