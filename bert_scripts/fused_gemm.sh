#!/bin/bash

# cmake -B debug -DCMAKE_BUILD_TYPE=Debug -DCUTLASS_NVCC_ARCHS=80  -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON .
# cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUTLASS_NVCC_ARCHS=80 -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON .

ncu_output=${1:-"report80"}

target=13_fused_two_gemms_s4_sm80_shmem

# cmake --build build --target $target
cd build
make -B $target

if [ $? -eq 0 ]; then
    echo "Compilation successful"
    cd ..
else
    echo "Compilation failed"
    cd ..
    exit 1
fi

# ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_rf
./build/examples/13_two_tensor_op_fusion/$target

# if [ "$ncu_output" = "0" ]; then
#     exit 0
# fi
# echo "output: $ncu_output.ncu-rep"
# ncu --cache-control all --clock-control base --target-processes all --rule SOLBottleneck --set full \
#     --nvtx --nvtx-include "Fused-GEMM/" --nvtx-include "Non-fused-GEMM/" --import-source on --call-stack \
#     -o $ncu_output.ncu-rep -f \
#     ./build/examples/13_two_tensor_op_fusion/$target
