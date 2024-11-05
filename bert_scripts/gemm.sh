# #!/bin/bash

# cmake --build build --target cutlass_test_unit_gemm_s4 -j &
# pid1=$!
# cmake --build build --target cutlass_test_unit_gemm_s8 -j &
# pid2=$!
# wait $pid1
# status1=$?
# wait $pid2
# status2=$?

# if [ $status1 -eq 0 ] && [ $status2 -eq 0 ]; then
#     echo "Compilation successful"
# else
#     echo "Compilation failed"
#     exit 1
# fi

# ./build/test/unit/gemm/device/cutlass_test_unit_gemm_s4
# ./build/test/unit/gemm/device/cutlass_test_unit_gemm_s8

# ncu --cache-control all --clock-control base --target-processes all --rule SOLBottleneck --set full \
#     --nvtx --nvtx-include "NCUTEST/" --import-source on --call-stack \
#     -o report-s4.ncu-rep -f \
#     ./build/test/unit/gemm/device/cutlass_test_unit_gemm_s4

# ncu --cache-control all --clock-control base --target-processes all --rule SOLBottleneck --set full \
#     --nvtx --nvtx-include "NCUTEST/" --import-source on --call-stack \
#     -o report-s4.ncu-rep -f \
#     ./build/test/unit/gemm/device/cutlass_test_unit_gemm_s8



cmake --build build --target 00_s8_gemm -j &
pid1=$!
wait $pid1
status1=$?

if [ $status1 -eq 0 ] ; then
    echo "Compilation successful"
else
    echo "Compilation failed"
    exit 1
fi

./build/examples/00_basic_gemm/00_s8_gemm

ncu --cache-control all --clock-control base --target-processes all --rule SOLBottleneck --set full \
    --nvtx --nvtx-include "NCUTEST/" --import-source on --call-stack \
    -o report-s8-persist.ncu-rep -f \
    ./build/examples/00_basic_gemm/00_s8_gemm