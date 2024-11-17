#!/bin/bash

parabuild . \
    examples/13_two_tensor_op_fusion/fused_two_gemms_s4_sm80_shmem.cu \
    build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s4_sm80_shmem \
    --workspaces-path /root/cutlass_workspace/workspaces \
    --in-place-template \
    --target 13_fused_two_gemms_s4_sm80_shmem \
    --init-cmake-args="-DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON" \
    --data-file "bert_scripts/sample.json" \
    --progress-bar \
    -j 46 \
    -J 1 \
    --cache \
    -o example.json

parabuild . \
    examples/13_two_tensor_op_fusion/fused_two_gemms_s4_sm80_shmem.cu \
    build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s4_sm80_shmem \
    --workspaces-path /root/cutlass_workspace/workspaces \
    --in-place-template \
    --target 13_fused_two_gemms_s4_sm80_shmem \
    --init-cmake-args="-DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON" \
    --data-file "bert_scripts/all.json" \
    --progress-bar \
    -j 46 \
    -J 1 \
    --cache \
    -o output.json

    # --data-file "bert_scripts/data.json" \

# cargo run --release \
#     .. \
#     examples/13_two_tensor_op_fusion/fused_two_gemms_s4_sm80_shmem.cu \
#     build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s4_sm80_shmem \
#     --workspaces-path /root/cutlass_workspace/workspaces \
#     --in-place-template \
#     --target 13_fused_two_gemms_s4_sm80_shmem \
#     --init-cmake-args="-DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON" \
#     --data '[{"N": 1}]' \
#     --progress-bar \
#     -j 1