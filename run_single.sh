#!/bin/bash
set -x

echo "Pipeline Perf evaluation"

# pip install absl

RUN_NAME=$1
NODES=$2
MBS=$3
ici_DP=$4
dcn_DP=$5
ici_FSDP=$6
ici_TP=$7
dcn_FSDP=$8
dcn_PP=$9
VP=${10}
AR_MULTIPLE=${11}
AG_MULTIPLE=${12}
RS_MULTIPLE=${13}
MEM_FRACTION=${14}
POLICY=${15}
QUANTIZATION=${16}

BASE_THRESHOLD=8589934592 # 8 GB

HLO_NAME="${RUN_NAME}-single-run-hlo"

HLO_DUMP_PATH="/opt/workspace/nsys_logs/${RUN_NAME}/$HLO_NAME"

export XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=1
export NCCL_IB_SL=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# export PYTHONFAULTHANDLER=1
# export TF_CPP_VMODULE=gpu_latency_hiding_scheduler=8,latency_hiding_scheduler=8,all_reduce_combiner=8
# export TF_CPP_VMODULE=gpu_executable=8,nccl_collectives=8,nccl_all_gather_thunk=8,nccl_all_reduce_thunk=8,nccl_all_to_all_thunk=8,nccl_api=8,nccl_api_stub=8,nccl_clique=8,nccl_collective_broadcast_thunk=8,nccl_collective_permute_thunk=8,nccl_collective_thunk=8,nccl_group_thunk=8,nccl_p2p_thunk_common=8,nccl_recv_thunk=8,nccl_send_thunk=8
# export TF_CPP_MIN_LOG_LEVEL=0
# export TF_CPP_MAX_LOG_LEVEL=10

# everything true
# export BASE_XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_enable_command_buffer=""
                --xla_gpu_all_reduce_combine_threshold_bytes=$((BASE_THRESHOLD/AR_MULTIPLE))
                --xla_gpu_all_gather_combine_threshold_bytes=$((BASE_THRESHOLD/AG_MULTIPLE))
                --xla_gpu_reduce_scatter_combine_threshold_bytes=$((BASE_THRESHOLD/RS_MULTIPLE))
                --xla_gpu_enable_pipelined_all_gather=true
                --xla_gpu_enable_pipelined_reduce_scatter=true
                --xla_gpu_enable_pipelined_all_reduce=true
                --xla_gpu_enable_while_loop_double_buffering=false
                --xla_gpu_enable_all_gather_combine_by_dim=false
                --xla_gpu_enable_reduce_scatter_combine_by_dim=false
                --xla_gpu_collective_permute_decomposer_threshold=0
                --xla_gpu_enable_pipelined_collectives=false
                --xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE
                --xla_disable_hlo_passes=rematerialization"
                # --xla_gpu_collective_permute_decomposer_threshold=0
                # --xla_gpu_enable_pipelined_collectives=false
                # --xla_gpu_experimental_pipeline_parallelism_opt_level=PIPELINE_PARALLELISM_OPT_LEVEL_ENABLE
                # --xla_dump_hlo_as_text
                # --xla_dump_to=$HLO_DUMP_PATH
                # --xla_dump_hlo_pass_re=.*"

echo "XLA_FLAGS = ${XLA_FLAGS}"
echo "XLA_PYTHON_CLIENT_MEM_FRACTION = ${XLA_PYTHON_CLIENT_MEM_FRACTION}"
echo "POLICY = ${POLICY}"

RUN_SETTINGS="maxtext/MaxText/train.py maxtext/MaxText/configs/base.yml run_name=${RUN_NAME}-single-run use_iota_embed=true scan_layers=true\
    steps=15 per_device_batch_size=${MBS} model_name=llama2-70b-8-layers remat_policy=${POLICY} enable_checkpointing=false logits_dot_in_fp32=false\
    base_output_directory=local_train dataset_path=local dataset_type=synthetic attention=cudnn_flash_te tokenizer_path=maxtext/assets/tokenizer.llama2\
    max_target_length=4096 quantization=${QUANTIZATION} hardware=gpu profiler=nsys skip_first_n_steps_for_profiler=9 profiler_steps=3\
    enable_goodput_recording=false monitor_goodput=false num_layers_per_pipeline_stage=$((8/(dcn_PP*VP)))\
    dcn_fsdp_parallelism=${dcn_FSDP} ici_fsdp_parallelism=${ici_FSDP}\
    ici_data_parallelism=${ici_DP} dcn_data_parallelism=${dcn_DP}\
    ici_tensor_parallelism=${ici_TP} dcn_tensor_parallelism=1\
    ici_pipeline_parallelism=${dcn_PP} dcn_pipeline_parallelism=1"
    # dcn_pipeline_parallelism=${dcn_PP}"

echo "SLURM_PROCID is: $SLURM_PROCID"
NSYS_OUTPUT_FILE="/opt/workspace/nsys_logs/${RUN_NAME}/${SLURM_PROCID}_${RUN_NAME}-single-run"

NSYS_CMD="nsys profile -s none -o ${NSYS_OUTPUT_FILE} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
# NSYS_CMD=""

echo "Command: $NSYS_CMD python3 $RUN_SETTINGS"

${NSYS_CMD} python3 ${RUN_SETTINGS}

set +x