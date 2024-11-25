#!/bin/bash

parabuild . \
    examples/13_two_tensor_op_fusion/fused_two_gemms_s4_sm80_shmem.cu \
    build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s4_sm80_shmem \
    --workspaces-path /root/cutlass_workspace/workspaces \
    --make-target 13_fused_two_gemms_s4_sm80_shmem \
    --init-cmake-args="-DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON" \
    --data-file "bert_scripts/sample.json" \
    -j 17 \
    -J 7 \
    -o bert_scripts/example.json

parabuild . \
    examples/13_two_tensor_op_fusion/fused_two_gemms_s4_sm80_shmem.cu \
    build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s4_sm80_shmem \
    --workspaces-path /root/cutlass_workspace/workspaces \
    --make-target 13_fused_two_gemms_s4_sm80_shmem \
    --init-cmake-args="-DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON" \
    --data-file "bert_scripts/fused.json" \
    -j 17 \
    -J 7 \
    -o bert_scripts/output-fused.json

parabuild . \
    examples/13_two_tensor_op_fusion/fused_two_gemms_s4_sm80_shmem.cu \
    build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s4_sm80_shmem \
    --workspaces-path /root/cutlass_workspace/workspaces \
    --make-target 13_fused_two_gemms_s4_sm80_shmem \
    --init-cmake-args="-DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON" \
    --data-file "bert_scripts/fused.json" \
    -j 22 \
    -J 2 \
    -o bert_scripts/output-fused.json