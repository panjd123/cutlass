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

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/host_reorder.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_relu.h"

#include "reference/device/tensor_scale_bias.h"
#include "helper.h"

#include <nvToolsExt.h>

#include "cutlass/coord.h"

using namespace cutlass;

using element_t = cutlass::int4b_t;

constexpr int STAGES = 3;

constexpr int INSTRUCTION_SHAPE_K = 64;
constexpr int FUSED_SHAPE_K = 128;

constexpr bool UseMajor = false;

constexpr int INTERLEAVE = UseMajor ? 0 : 64;
constexpr bool SMEM_ACCUMULATOR = true;

#define TESTM 12 * 256

#ifndef TESTK
#define TESTK 384
#endif

#ifndef PARABUILD
#define TESTN1 384
#define TESTN2 384
#else
#define TESTN1         \
  {                    \
    {                  \
      default N1 "384" \
    }                  \
  }
#define TESTN2         \
  {                    \
    {                  \
      default N2 "384" \
    }                  \
  }
#endif

// K -> N1 -> N2
// 384 -> 384 -> 384*4 -> 384

////////////////////////////////////////////////////////////////////////////////

cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0_0(TESTM, TESTN1, TESTK);
cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0_1(TESTM, TESTN2, TESTN1);

bool run_nonfused_gemm_s8_sm80()
{

  using ElementOutput = element_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(0); // beta=1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(0); // beta=1 for bias

#ifndef PARABUILD
  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 192, 128>;
  using WarpShape0 = cutlass::gemm::GemmShape<128, 32, 128>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 192, 128>;
  using WarpShape1 = cutlass::gemm::GemmShape<128, 32, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
#else
  using ThreadblockShape0 = cutlass::gemm::GemmShape<{{default NFThreadblockShapeM "128"}}, {{default NFThreadblockShapeN "192"}}, {{default NFThreadblockShapeK "128"}}>;
  using WarpShape0 = cutlass::gemm::GemmShape<{{default NFWarpShapeM "128"}}, {{default NFWarpShapeN "32"}}, {{default NFWarpShapeK "128"}}>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<{{default NFThreadblockShapeM "128"}}, {{default NFThreadblockShapeN "192"}}, {{default NFThreadblockShapeK "128"}}>;
  using WarpShape1 = cutlass::gemm::GemmShape<{{default NFWarpShapeM "128"}}, {{default NFWarpShapeN "32"}}, {{default NFWarpShapeK "128"}}>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  printf("ThreadblockShape0: %d %d %d\n", ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK);
  printf("WarpShape0: %d %d %d\n", WarpShape0::kM, WarpShape0::kN, WarpShape0::kK);
  printf("ThreadblockShape1: %d %d %d\n", ThreadblockShape1::kM, ThreadblockShape1::kN, ThreadblockShape1::kK);
  printf("WarpShape1: %d %d %d\n", WarpShape1::kM, WarpShape1::kN, WarpShape1::kK);
#endif

  using Gemm0 = cutlass::gemm::device::Gemm<
      element_t,
      std::conditional<UseMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajorInterleaved<INTERLEAVE>>::type,
      element_t,
      std::conditional<UseMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajorInterleaved<INTERLEAVE>>::type,
      ElementOutput,
      std::conditional<UseMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajorInterleaved<INTERLEAVE>>::type,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape0,
      WarpShape0,
      InstructionShape,
      cutlass::epilogue::thread::LinearCombinationRelu<
          ElementOutput,
          64 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementCompute,
          cutlass::epilogue::thread::ScaleType::Nothing>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      STAGES,
      32,
      32,
      false,
      cutlass::arch::OpMultiplyAddSaturate>;
  using Gemm1 = cutlass::gemm::device::Gemm<
      element_t,
      std::conditional<UseMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajorInterleaved<INTERLEAVE>>::type,
      element_t,
      std::conditional<UseMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajorInterleaved<INTERLEAVE>>::type,
      ElementOutput,
      std::conditional<UseMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajorInterleaved<INTERLEAVE>>::type,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape1,
      WarpShape1,
      InstructionShape,
      cutlass::epilogue::thread::LinearCombinationRelu<
          ElementOutput,
          64 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementCompute,
          cutlass::epilogue::thread::ScaleType::Nothing>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      STAGES,
      32,
      32,
      false,
      cutlass::arch::OpMultiplyAddSaturate>;

  B2bInterleavedNonFusedGemmRun<Gemm0, Gemm1, INTERLEAVE> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back INT4 NT interleaved GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_s8_sm80_problem_size_0_0, gemm_s8_sm80_problem_size_0_1, alpha0, beta0, alpha1, beta1);
  if (pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
  // return 1;
}

bool run_fused_gemm_s8_sm80_shmem()
{

  using ElementOutput = element_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  // Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0);
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(0); // beta=1 for bias

#ifndef PARABUILD
  using ThreadblockShape0 = cutlass::gemm::GemmShape<192, 384, FUSED_SHAPE_K>;
  using WarpShape0 = cutlass::gemm::GemmShape<192, 48, FUSED_SHAPE_K>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<192, 384, FUSED_SHAPE_K>;
  using WarpShape1 = cutlass::gemm::GemmShape<192, 48, FUSED_SHAPE_K>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, INSTRUCTION_SHAPE_K>;
#else
  using ThreadblockShape0 = cutlass::gemm::GemmShape<{{default FThreadblockShapeM "192"}}, {{default FThreadblockShapeN "TESTN1"}}, {{default FThreadblockShapeK "FUSED_SHAPE_K"}}>;
  using WarpShape0 = cutlass::gemm::GemmShape<{{default FWarpShapeM "192"}}, {{default FWarpShapeN "48"}}, {{default FWarpShapeK "FUSED_SHAPE_K"}}>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<{{default FThreadblockShapeM "192"}}, {{default FThreadblockShapeN "TESTN2"}}, {{default FThreadblockShapeK "FUSED_SHAPE_K"}}>;
  using WarpShape1 = cutlass::gemm::GemmShape<{{default FWarpShapeM "192"}}, {{default FWarpShapeN "48"}}, {{default FWarpShapeK "FUSED_SHAPE_K"}}>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  printf("ThreadblockShape0: %d %d %d\n", ThreadblockShape0::kM, ThreadblockShape0::kN, ThreadblockShape0::kK);
  printf("WarpShape0: %d %d %d\n", WarpShape0::kM, WarpShape0::kN, WarpShape0::kK);
  printf("ThreadblockShape1: %d %d %d\n", ThreadblockShape1::kM, ThreadblockShape1::kN, ThreadblockShape1::kK);
  printf("WarpShape1: %d %d %d\n", WarpShape1::kM, WarpShape1::kN, WarpShape1::kK);
#endif

  using EpilogueOutputOp0 =
      cutlass::epilogue::thread::LinearCombinationRelu<
          ElementOutput,
          // 8 * InstructionShape::kN / 32,
          64 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementCompute,
          cutlass::epilogue::thread::ScaleType::Nothing>;

  using EpilogueOutputOp1 =
      cutlass::epilogue::thread::LinearCombinationRelu<
          ElementOutput,
          64 / cutlass::sizeof_bits<ElementOutput>::value,
          ElementAccumulator,
          ElementCompute,
          cutlass::epilogue::thread::ScaleType::Nothing>;

  using B2bGemm = cutlass::gemm::device::B2bGemm<
      element_t,
      std::conditional<UseMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajorInterleaved<INTERLEAVE>>::type,
      element_t,
      std::conditional<UseMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajorInterleaved<INTERLEAVE>>::type,
      ElementOutput,
      std::conditional<UseMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajorInterleaved<INTERLEAVE>>::type,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      ThreadblockShape0,
      ThreadblockShape1,
      WarpShape0,
      WarpShape1,
      InstructionShape,
      EpilogueOutputOp0,
      EpilogueOutputOp1,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      STAGES,
      SMEM_ACCUMULATOR,
      32,
      32,
      cutlass::arch::OpMultiplyAddSaturate>;

  B2bInterleavedFusedGemmRun<B2bGemm, INTERLEAVE> fusedGemm;

  std::cout << "Running Fused back-to-back INT4 NT interleaved GEMMs with shared memory staging...\n";
  bool passed = fusedGemm.run(
      gemm_s8_sm80_problem_size_0_0,
      gemm_s8_sm80_problem_size_0_1,
      alpha0,
      beta0,
      alpha1,
      beta1);

  if (passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
  // return 1;
}

int main()
{
  int device_count;
  cudaGetDeviceCount(&device_count);
  printf("系统中可用的 GPU 数量: %d\n", device_count);
  int device_id = device_count - 1;
  const char *parabuild_id = std::getenv("PARABUILD_ID");
  if (parabuild_id)
  {
    device_id = std::atoi(parabuild_id);
  }
  device_id = device_id % device_count;
  printf("使用的 GPU 设备 ID: %d\n", device_id);
  CUDA_CHECK(cudaSetDevice(device_id));
  ;

  //
  // Determine SMEM requirements and waive if not satisfied
  //
  cudaDeviceProp properties;
  int device_idx;
  cudaError_t result = cudaGetDevice(&device_idx);

  if (result != cudaSuccess)
  {
    throw std::runtime_error("cudaGetDevice() API call failed.");
  }

  result = cudaGetDeviceProperties(&properties, device_idx);

  if (result != cudaSuccess)
  {
    throw std::runtime_error("cudaGetDeviceProperties() failed");
  }

  std::cout << "Device: " << properties.name << std::endl;
  std::cout << "Arch: SM" << properties.major << properties.minor << std::endl;
  std::cout << "UUID: ";
  for (int i = 0; i < 16; i++)
  {
    std::cout << std::hex << static_cast<int>(properties.uuid.bytes[i]);
  }
  std::cout << std::dec << std::endl;

  std::cout << properties.multiProcessorCount << " SMs" << std::endl;
  std::cout << properties.l2CacheSize / 1024 << " KB of L2 cache" << std::endl;

  std::cout << properties.sharedMemPerBlock / 1024 << " KB of shared memory per block" << std::endl;
  std::cout << properties.sharedMemPerMultiprocessor / 1024 << " KB of shared memory per SM" << std::endl;

  std::cout << properties.maxThreadsPerBlock << " threads per block" << std::endl;
  std::cout << properties.maxThreadsPerMultiProcessor << " threads per SM" << std::endl;

  std::cout << properties.regsPerBlock << " registers per block" << std::endl;
  std::cout << properties.regsPerMultiprocessor << " registers per block" << std::endl;

  std::cout << properties.totalConstMem / 1024 << " KB of constant memory" << std::endl;
  std::cout << properties.totalGlobalMem / 1024 / 1024 << " MB of global memory" << std::endl;

  // std::vector<bool (*)()>funcs = {
  //   // &run_fused_gemm_s8_sm80_shmem,
  //   &run_nonfused_gemm_s8_sm80
  // };

  // return testRun(80, funcs, "gemm int4 RF residency");

  constexpr int interleave = 16;
  using element_t = int4b_t;
  using ElementA = element_t;
  using ElementB = element_t;
  using ElementC = element_t;

  using ElementOutput = element_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  using Operator = typename cutlass::gemm::device::DefaultGemmConfiguration<
        typename cutlass::arch::OpClassTensorOp, arch::Sm80, ElementA, ElementB, ElementC,
        ElementAccumulator>::Operator;

  using RLayout = cutlass::layout::RowMajorInterleaved<interleave>;
  using CLayout = cutlass::layout::ColumnMajorInterleaved<interleave>;

  cutlass::gemm::GemmCoord problem_size_0(32, 32, 32);

  using ThreadblockShape0 = cutlass::gemm::GemmShape<192, 384, 128>;
  using WarpShape0 = cutlass::gemm::GemmShape<192, 48, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;

  using Gemm0 = cutlass::gemm::device::Gemm<
    element_t,
    CLayout,
    element_t,
    RLayout,
    ElementOutput,
    CLayout,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape0,
    WarpShape0,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Nothing
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    STAGES,
    32,
    32,
    false,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  cutlass::HostTensor<
      Gemm0::ElementA,
      Gemm0::LayoutA>
      tensor_A0(problem_size_0.mk());

  cutlass::HostTensor<
      Gemm0::ElementB,
      Gemm0::LayoutB>
      tensor_B0(problem_size_0.kn());

  cutlass::HostTensor<
      Gemm0::ElementB,
      Gemm0::LayoutB>
      tensor_B0_reordered(problem_size_0.kn());

  cutlass::HostTensor<
      Gemm0::ElementC,
      Gemm0::LayoutC>
      tensor_C0(problem_size_0.mn());

  cutlass::HostTensor<
      Gemm0::ElementC,
      Gemm0::LayoutC>
      tensor_D0(problem_size_0.mn());

  cutlass::HostTensor<
      Gemm0::ElementB,
      Gemm0::LayoutC>
      tensor_Bias0({1, problem_size_0.n()});

  cutlass::reference::host::BlockFillSequentialModN(
      tensor_A0.host_data(),
      tensor_A0.capacity(),
      3);

  cutlass::reference::host::BlockFillSequentialModN(
      tensor_B0.host_data(),
      tensor_B0.capacity(),
      3);

  cutlass::reference::host::BlockFillSequentialModN(
      tensor_C0.host_data(),
      tensor_C0.capacity(),
      3);

  cutlass::reference::host::BlockFillSequentialModN(
      tensor_Bias0.host_data(),
      tensor_Bias0.capacity(),
      3);

  cutlass::reference::host::Gemm<
      Gemm0::ElementA, Gemm0::LayoutA,
      Gemm0::ElementB, Gemm0::LayoutB,
      Gemm0::ElementC, Gemm0::LayoutC, ElementCompute,
      ElementAccumulator, Operator>
      gemm;

  ElementCompute alpha = ElementCompute(1);
  ElementCompute beta = ElementCompute(0);
  gemm(
      problem_size_0,
      alpha, 
      tensor_A0.host_ref(), 
      tensor_B0.host_ref(), 
      beta, 
      tensor_D0.host_ref()
  );

  // dump
  std::ofstream file("tensor_A0.txt");
  for(int i = 0; i <problem_size_0.m(); i++){
    for(int j = 0; j < problem_size_0.k(); j++){
      file << static_cast<int>(tensor_A0.at({i, j})) << " ";
    }
    file << std::endl;
  }
  std::ofstream file1("tensor_B0.txt");
  for(int i = 0; i <problem_size_0.k(); i++){
    for(int j = 0; j < problem_size_0.n(); j++){
      file1 << static_cast<int>(tensor_B0.at({i, j})) << " ";
    }
    file1 << std::endl;
  }

  for(int i = 0; i <problem_size_0.m(); i++){
    for(int j = 0; j < problem_size_0.n(); j++){
      std::cout << static_cast<int>(tensor_D0.at({i, j})) << " ";
    }
    std::cout << std::endl;
  }

  cutlass::reorder_column<interleave>(
    tensor_B0_reordered.host_ref(), tensor_B0.host_ref(), problem_size_0);
  tensor_A0.sync_device();
  tensor_B0.sync_device();
  tensor_B0_reordered.sync_device();
  tensor_C0.sync_device();
  tensor_D0.sync_device();
  tensor_Bias0.sync_device();

  
  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(0);

  Gemm0::Arguments arguments_0{
  problem_size_0,
  tensor_A0.device_ref(),
  tensor_B0_reordered.device_ref(),
  {tensor_Bias0.device_data(), Gemm0::LayoutC::Stride(0)},
  tensor_D0.device_ref(),
  {alpha0, beta0}
  };

  Gemm0 gemm_op_0;

  cutlass::Status status = gemm_op_0.initialize(arguments_0);
};

////////////////////////////////////////////////////////////////////////////////
