#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/gemm.h"

#include "device/b2b_gemm.h"
#include "b2b_interleaved_gemm_run.h"
#include "test_run.h"

using element_t = cutlass::int4b_t;
using ElementCompute = float;

int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);
  printf("系统中可用的 GPU 数量: %d\n", device_count);

  CUDA_CHECK(cudaSetDevice(device_count - 1));
  
//   element_t a = 1;
//   element_t b = 2;
//   element_t c = a + b;
//   std::cout << "c = " << c << std::endl;
  element_t a(1);
  element_t b(2);
  element_t c(a + b);
  std::cout << "c = " << c << std::endl;

  cutlass::HostTensor<element_t, cutlass::layout::RowMajor> tensor_a({2, 3});
  cutlass::HostTensor<element_t, cutlass::layout::RowMajor> tensor_b({3, 2});
  cutlass::HostTensor<element_t, cutlass::layout::RowMajor> tensor_c({2, 2});
  for (int i = 0; i < tensor_a.capacity(); ++i) {
    tensor_a.host_data()[i] = element_t(i);
    tensor_b.host_data()[i] = element_t(i);
  }
  cutlass::reference::host::TensorFill(tensor_c.host_view(), element_t(0));
}