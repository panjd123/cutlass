#!/bin/bash

# cmake -B debug -DCMAKE_BUILD_TYPE=Debug -DCUTLASS_NVCC_ARCHS=80  -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON .
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON .

cmake --build build --target 13_fused_two_gemms_s8_sm80_shmem

if [ $? -eq 0 ]; then
    echo "Compilation successful"
else
    echo "Compilation failed"
    exit 1
fi

# ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_rf
./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_shmem
ncu --cache-control all --clock-control base --target-processes all --rule SOLBottleneck --set full \
    --nvtx --nvtx-include "Fused-GEMM/" --nvtx-include "Non-fused-GEMM/" --import-source on --call-stack \
    -o report80.ncu-rep -f \
    ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_shmem