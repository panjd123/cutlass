#!/bin/bash

# cmake -B debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=80 -DCUTLASS_ENABLE_CUBLAS=ON -DCUTLASS_ENABLE_CUDNN=ON .

cmake --build build --target 13_fused_two_gemms_s8_sm80_rf -j &
pid1=$!
cmake --build build --target 13_fused_two_gemms_s8_sm80_shmem -j &
pid2=$!

wait $pid1
status1=$?

wait $pid2
status2=$?

if [ $status1 -eq 0 ] && [ $status2 -eq 0 ]; then
    echo "Build successful"
else
    echo "Build failed"
    exit 1
fi

# ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_rf
./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_shmem
