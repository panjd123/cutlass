#!/bin/bash

cmake --build build --target cutlass_test_unit_gemm_s4 -j &
cmake --build build --target cutlass_test_unit_gemm_s8 -j &
wait

./build/test/unit/gemm/device/cutlass_test_unit_gemm_s4
./build/test/unit/gemm/device/cutlass_test_unit_gemm_s8
# ./build/examples/00_basic_gemm/00_basic_gemm 3072 384 384
# ./build/examples/00_basic_gemm/00_s8_gemm 3072 384 384