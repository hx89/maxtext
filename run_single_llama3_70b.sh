#!/bin/bash
set -x

# pip install absl

export CODE_DIR=${CODE_DIR:-"."}
export LOG_DIR=${LOG_DIR:-"."}

num_nodes=2
num_gpus=$((8*num_nodes))

echo "num_nodes: $num_nodes, num_gpus: $num_gpus"

ici_DP=8
dcn_DP=$num_nodes
ici_FSDP=1
dcn_FSDP=1

HLO_DUMP_PATH="xla_dump"
BASE_THRESHOLD=8589934592 # 8 GB
RS_MULTIPLE=$num_gpus
AR_MULTIPLE=$num_gpus
AG_MULTIPLE=$num_gpus

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NCCL_IB_SL=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# everything true
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_enable_command_buffer=
                --xla_gpu_all_reduce_combine_threshold_bytes=$((BASE_THRESHOLD/AR_MULTIPLE))
                --xla_gpu_all_gather_combine_threshold_bytes=$((BASE_THRESHOLD/AG_MULTIPLE))
                --xla_gpu_reduce_scatter_combine_threshold_bytes=$((BASE_THRESHOLD/RS_MULTIPLE))
                --xla_gpu_enable_pipelined_all_gather=true
                --xla_gpu_enable_pipelined_reduce_scatter=true
                --xla_gpu_enable_pipelined_all_reduce=true
                --xla_gpu_enable_while_loop_double_buffering=false
                --xla_gpu_enable_all_gather_combine_by_dim=false
                --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_dump_hlo_as_text
                --xla_dump_to=$HLO_DUMP_PATH
                --xla_disable_hlo_passes=rematerialization
                --xla_gpu_graph_level=0"

echo "XLA_FLAGS = ${XLA_FLAGS}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION = ${XLA_PYTHON_CLIENT_MEM_FRACTION}"

# GA=4
# RUN_SETTINGS="-m MaxText.train MaxText/configs/base.yml run_name=debug_run base_output_directory=./debug_logs hardware=gpu dataset_type=synthetic model_name=llama3.3-70b remat_policy='minimal' scan_layers=False attention='cudnn_flash_jax' steps=20 dtype=bfloat16 max_target_length=8192 per_device_batch_size=4 ici_data_parallelism=${ici_DP} dcn_data_parallelism=${dcn_DP} ici_fsdp_parallelism=${ici_FSDP} dcn_fsdp_parallelism=${dcn_FSDP} profiler=nsys enable_checkpointing=false override_model_config=True base_num_decoder_layers=2 gradient_accumulation_steps=4"

RUN_SETTINGS="-m MaxText.train MaxText/configs/base.yml run_name=debug_run base_output_directory=./debug_logs hardware=gpu_multiprocess dataset_type=synthetic model_name=llama3-70b remat_policy='minimal' scan_layers=False attention='cudnn_flash_jax' steps=10 dtype=bfloat16 max_target_length=8192 per_device_batch_size=2 ici_data_parallelism=${ici_DP} dcn_data_parallelism=${dcn_DP} ici_fsdp_parallelism=${ici_FSDP} dcn_fsdp_parallelism=${dcn_FSDP} profiler=nsys enable_checkpointing=false override_model_config=True gradient_accumulation_steps=8 profiler_steps=2 shard_optimizer_over_data=True base_num_decoder_layers=16"

# Real dataset
# RUN_SETTINGS="-m MaxText.train MaxText/configs/base.yml run_name=debug_run base_output_directory=./debug_logs hardware=gpu dataset_path=/data/c4/mlperf-tfrecord dataset_name=c4/en:3.1.0 model_name=llama3-8b remat_policy='minimal' scan_layers=False attention='cudnn_flash_jax' steps=10 dtype=bfloat16 max_target_length=8192 per_device_batch_size=2 ici_data_parallelism=${ici_DP} dcn_data_parallelism=${dcn_DP} ici_fsdp_parallelism=${ici_FSDP} dcn_fsdp_parallelism=${dcn_FSDP} profiler=nsys enable_checkpointing=false override_model_config=True gradient_accumulation_steps=8 profiler_steps=2 shard_optimizer_over_data=True base_num_decoder_layers=2"

echo "SLURM_PROCID is: $SLURM_PROCID"
NSYS_OUTPUT_FILE="${LOG_DIR_LOCAL}/output-profile"

NSYS_CMD="nsys profile -s none -o ${NSYS_OUTPUT_FILE} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"

echo "Command: $NSYS_CMD python3 $RUN_SETTINGS"
${NSYS_CMD} python3 ${RUN_SETTINGS}

set +x
