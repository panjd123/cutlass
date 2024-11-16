/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
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

using element_t = int8_t;

constexpr int STAGES = 3;
constexpr int FUSED_SHAPE_K = 32;

constexpr bool UseMajor = false;

constexpr int INTERLEAVE = UseMajor ? 0 : 32;
constexpr bool SMEM_ACCUMULATOR = true;

#define TESTM 12*256

#ifndef TESTK
#define TESTK 384
#endif

#ifndef PROFILING
#define TESTN1 320
#define TESTN2 320
#else
#define TESTN1 {{N1}}
#define TESTN2 {{N2}}
#endif

// K -> N1 -> N2
// 384 -> 384 -> 384*4 -> 384

////////////////////////////////////////////////////////////////////////////////

cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_0(TESTM, TESTN1, TESTK);
cutlass::gemm::GemmCoord gemm_s8_sm80_problem_size_1(TESTM, TESTN2, TESTN1);

bool run_nonfused_gemm_s8_sm80() {

  using ElementOutput = element_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  ElementCompute beta0 = ElementCompute(0); //beta=1 for bias
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(0); //beta=1 for bias

  // using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 96, 64>;
  // using WarpShape0 = cutlass::gemm::GemmShape<32, 96, 64>;
  // using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 96, 64>;
  // using WarpShape1 = cutlass::gemm::GemmShape<32, 96, 64>;
  // using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
#ifndef PROFILING
  using ThreadblockShape0 = cutlass::gemm::GemmShape<128, 96, 64>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, 96, 64>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<128, 96, 64>;
  using WarpShape1 = cutlass::gemm::GemmShape<32, 96, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
#else
  using ThreadblockShape0 = cutlass::gemm::GemmShape<{{NFThreadblockShapeM}}, {{NFThreadblockShapeN}}, {{NFThreadblockShapeK}}>;
  using WarpShape0 = cutlass::gemm::GemmShape<{{NFWarpShapeM}}, {{NFWarpShapeN}}, {{NFWarpShapeK}}>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<{{NFThreadblockShapeM}}, {{NFThreadblockShapeN}}, {{NFThreadblockShapeK}}>;
  using WarpShape1 = cutlass::gemm::GemmShape<{{NFWarpShapeM}}, {{NFWarpShapeN}}, {{NFWarpShapeK}}>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
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
      cutlass::epilogue::thread::ScaleType::Nothing
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    STAGES,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddSaturate
  >;
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
      cutlass::epilogue::thread::ScaleType::Nothing
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    STAGES,
    16,
    16,
    false,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedNonFusedGemmRun<Gemm0, Gemm1, INTERLEAVE> nonFusedGemm;

  std::cout << "Running Non-fused back-to-back INT8 NT interleaved GEMMs...\n";
  bool pass = nonFusedGemm.run(gemm_s8_sm80_problem_size_0, gemm_s8_sm80_problem_size_1, alpha0, beta0, alpha1, beta1);
  if(pass)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return pass;
}

bool run_fused_gemm_s8_sm80_shmem() {

  using ElementOutput = element_t;
  using ElementAccumulator = int32_t;
  using ElementCompute = float;

  ElementCompute alpha0 = ElementCompute(1);
  //Fused kernel has built-in bias, setting beta=0
  ElementCompute beta0 = ElementCompute(0);
  ElementCompute alpha1 = ElementCompute(1);
  ElementCompute beta1 = ElementCompute(0); //beta=1 for bias

  // using ThreadblockShape0 = cutlass::gemm::GemmShape<32, 384, 64>;
  // using WarpShape0 = cutlass::gemm::GemmShape<32, 96, 64>;
  // using ThreadblockShape1 = cutlass::gemm::GemmShape<32, 384, 64>;
  // using WarpShape1 = cutlass::gemm::GemmShape<32, 96, 64>;
  // using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
#ifndef PROFILING
  using ThreadblockShape0 = cutlass::gemm::GemmShape<32, TESTN1, FUSED_SHAPE_K>;
  using WarpShape0 = cutlass::gemm::GemmShape<32, TESTN1 / 4, FUSED_SHAPE_K>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<32, TESTN2, FUSED_SHAPE_K>;
  using WarpShape1 = cutlass::gemm::GemmShape<32, TESTN2 / 4, FUSED_SHAPE_K>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
#else
  using ThreadblockShape0 = cutlass::gemm::GemmShape<{{FThreadblockShapeM}}, {{FThreadblockShapeN}}, {{FThreadblockShapeK}}>;
  using WarpShape0 = cutlass::gemm::GemmShape<{{FWarpShapeM}}, {{FWarpShapeN}}, {{FWarpShapeK}}>;
  using ThreadblockShape1 = cutlass::gemm::GemmShape<{{FThreadblockShapeM}}, {{FThreadblockShapeN}}, {{FThreadblockShapeK}}>;
  using WarpShape1 = cutlass::gemm::GemmShape<{{FWarpShapeM}}, {{FWarpShapeN}}, {{FWarpShapeK}}>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
#endif

  using EpilogueOutputOp0 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      8 * InstructionShape::kN / 32,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Nothing
    >;

  using EpilogueOutputOp1 =
    cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      64 / cutlass::sizeof_bits<ElementOutput>::value,
      ElementAccumulator,
      ElementCompute,
      cutlass::epilogue::thread::ScaleType::Nothing
    >;

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
    16,
    16,
    cutlass::arch::OpMultiplyAddSaturate
  >;

  B2bInterleavedFusedGemmRun<B2bGemm, INTERLEAVE> fusedGemm;

  std::cout << "Running Fused back-to-back INT8 NT interleaved GEMMs with shared memory staging...\n";
  bool passed = fusedGemm.run(
    gemm_s8_sm80_problem_size_0,
    gemm_s8_sm80_problem_size_1,
    alpha0,
    beta0,
    alpha1,
    beta1
    );

  if(passed)
    std::cout << "Pass\n";
  else
    std::cout << "Fail\n";

  return passed;
}


int main() {
  int device_count;
  cudaGetDeviceCount(&device_count);
  printf("系统中可用的 GPU 数量: %d\n", device_count);

  CUDA_CHECK(cudaSetDevice(device_count - 1));
  std::vector<bool (*)()>funcs = {
    &run_fused_gemm_s8_sm80_shmem,
    &run_nonfused_gemm_s8_sm80
  };

  return testRun(80, funcs, "gemm int8 RF residency");
}

////////////////////////////////////////////////////////////////////////////////
