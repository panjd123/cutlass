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

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutlass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

using data_type = int8_t;
using ElementOutput = int32_t;
using ElementCompute = float;
using ElementAccumulator = int32_t;

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  data_type alpha,
  data_type const *A,
  int lda,
  data_type const *B,
  int ldb,
  data_type beta,
  ElementOutput *C,
  int ldc,
  cudaStream_t stream = nullptr
  ) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  // using CutlassGemm = cutlass::gemm::device::Gemm<data_type,        // Data-type of A matrix
  //                                                 cutlass::layout::RowMajor,  // Layout of A matrix
  //                                                 data_type,        // Data-type of B matrix
  //                                                 ColumnMajor,  // Layout of B matrix
  //                                                 data_type,        // Data-type of C matrix
  //                                                 ColumnMajor>; // Layout of C matrix

  // cutlass::arch::OpClassTensorOp

  using CutlassGemm = cutlass::gemm::device::Gemm<
      data_type, cutlass::layout::RowMajor, data_type, cutlass::layout::ColumnMajor,
      ElementOutput, cutlass::layout::ColumnMajor, ElementAccumulator,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
      cutlass::epilogue::thread::LinearCombinationClamp<
          ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementCompute>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
    
  // using CutlassGemm = cutlass::gemm::device::Gemm<
  //     data_type, cutlass::layout::RowMajor, data_type, cutlass::layout::ColumnMajor,
  //     ElementOutput, cutlass::layout::ColumnMajor, ElementAccumulator,
  //     cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
  //     cutlass::gemm::GemmShape<128, 96, 64>,
  //     cutlass::gemm::GemmShape<32, 96, 64>, cutlass::gemm::GemmShape<16, 8, 32>,
  //     cutlass::epilogue::thread::LinearCombinationClamp<
  //         ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value, ElementAccumulator, ElementCompute>,
  //     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 3>;
      

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
                              {C, ldc},    // Tensor-ref for source matrix C
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                              {alpha, beta}); // Scalars used in the Epilogue

  //
  // Launch the CUTLASS GEMM kernel.
  //

  cutlass::Status status = gemm_operator(args, nullptr, stream);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
template <typename data_type>
__global__ void InitializeMatrix_kernel(
  data_type *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * rows;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 4;
    data_type value = data_type(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template <typename data_type>
cudaError_t InitializeMatrix(data_type *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
template <typename data_type>
cudaError_t AllocateMatrix(data_type **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(data_type) * rows * columns;

  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix(*matrix, rows, columns, seed);

  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  data_type alpha,
  data_type const *A,
  int lda,
  data_type const *B,
  int ldb,
  data_type beta,
  ElementOutput *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    ElementAccumulator accumulator = 0;

    for (int k = 0; k < K; ++k) {
      // accumulator += A[i + k * lda] * B[k + j * ldb];
      // A RowMajor, B ColumnMajor
      accumulator += static_cast<ElementCompute>(A[k + i * lda]) * static_cast<ElementCompute>(B[k + j * ldb]);
    }

    C[i + j * ldc] = static_cast<ElementOutput>(alpha * accumulator + beta * C[i + j * ldc]);
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  data_type alpha,
  data_type const *A,
  int lda,
  data_type const *B,
  int ldb,
  data_type beta,
  ElementOutput *C,
  int ldc,
  cudaStream_t stream = nullptr
  ) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block, 0, stream >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, data_type alpha, data_type beta, int num_iterations = 10000) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(ElementOutput) * ldc * N;

  // Define pointers to matrices in GPU device memory.
  data_type *A;
  data_type *B;
  ElementOutput *C_cutlass;
  ElementOutput *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix(&A, M, K, 0);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(A);
    return result;
  }

  result = AllocateMatrix(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  result = AllocateMatrix(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }

  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  int num_warmup = std::max(1, num_iterations / 10);

  std::cout << "A:" << reinterpret_cast<const void*>(A) << "\t" << "B:" << reinterpret_cast<const void*>(B) << "\t" 
  << "C_cutlass:" << reinterpret_cast<void*>(C_cutlass) << "\t" << "C_reference:" << reinterpret_cast<void*>(C_reference) << std::endl;
  size_t persisting_start = 0;
  persisting_start = min(reinterpret_cast<size_t>(A), reinterpret_cast<size_t>(B));
  persisting_start = min(persisting_start, reinterpret_cast<size_t>(C_cutlass));
  persisting_start = min(persisting_start, reinterpret_cast<size_t>(C_reference));
  void* persisting_ptr = reinterpret_cast<void*>(persisting_start);

  size_t persisting_end = 0;
  persisting_end = max(reinterpret_cast<size_t>(A) + sizeof(data_type) * M * K, reinterpret_cast<size_t>(B) + sizeof(data_type) * K * N);
  persisting_end = max(persisting_end, reinterpret_cast<size_t>(C_cutlass) + sizeof(ElementOutput) * M * N);
  persisting_end = max(persisting_end, reinterpret_cast<size_t>(C_reference) + sizeof(ElementOutput) * M * N);
  size_t persisting_size = persisting_end - persisting_start;
  
  // return cudaSuccess;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // size_t size = min(int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize);
  // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);
  size_t window_size = min(prop.accessPolicyMaxWindowSize, static_cast<int>(sizeof(data_type) * N * K));
  cudaStreamAttrValue stream_attribute;                                                       // Stream level attributes data structure
  // stream_attribute.accessPolicyWindow.base_ptr  = const_cast<void*>(reinterpret_cast<const void*>(B));               // Global Memory data pointer
  stream_attribute.accessPolicyWindow.base_ptr  = persisting_ptr;               // Global Memory data pointer
  // stream_attribute.accessPolicyWindow.num_bytes = window_size;                                // Number of bytes for persistence access
  stream_attribute.accessPolicyWindow.num_bytes = persisting_size;                                // Number of bytes for persistence access
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persisting_size);
  stream_attribute.accessPolicyWindow.hitRatio  = 1;                                          // Hint for cache hit ratio
  stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;               // Persistence Property
  stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;                // Type of access property on cache miss

  std::cout << "l2CacheSize:" << prop.l2CacheSize << "\t" << "persistingL2CacheMaxSize:" << prop.persistingL2CacheMaxSize << "\t" << "accessPolicyMaxWindowSize:" << prop.accessPolicyMaxWindowSize << std::endl;

  std::cout << "CUTLASS GEMM kernel (ms): " << std::endl;
  std::vector<float> ms_cutlass_history;
  for (int i = 0; i < num_warmup + num_iterations; ++i) {
    auto start_cutlass = std::chrono::high_resolution_clock::now();
    if(i == num_warmup - 1){
      nvtxRangePushA("NCUTEST");
    }
    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, stream);
    if(i == num_warmup - 1){
      nvtxRangePop();
    }
    auto end_cutlass = std::chrono::high_resolution_clock::now();
    float ms_cutlass = std::chrono::duration<float>(end_cutlass - start_cutlass).count() * 1000;
    if (i >= num_warmup){
      // std::cout << ms_cutlass << " ";
      ms_cutlass_history.push_back(ms_cutlass);
    }

    if (result != cudaSuccess) {
      std::cerr << "CUTLASS GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }
  }
  quantile(ms_cutlass_history);
  // std::cout << std::endl;
  

  //
  // Verify.
  //

  // Launch reference GEMM
  std::vector<float> ms_reference_history;
  std::cout << "Reference GEMM kernel (ms): " << std::endl;
  for (int i = 0; i < num_iterations; ++i) {
    auto start_reference = std::chrono::high_resolution_clock::now();
    if(i == num_warmup - 1){
      nvtxRangePushA("NCUTEST");
    }
    result = ReferenceGemm(M, N, K, alpha, A, lda, B, ldb, beta, C_reference, ldc, stream);
    if(i == num_warmup - 1){
      nvtxRangePop();
    }
    auto end_reference = std::chrono::high_resolution_clock::now();
    float ms_reference = std::chrono::duration<float>(end_reference - start_reference).count() * 1000;
    if (i >= num_warmup){
      // std::cout << ms_reference << " ";
      ms_reference_history.push_back(ms_reference);
    }

    if (result != cudaSuccess) {
      std::cerr << "Reference GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }
  }
  quantile(ms_reference_history);
  // std::cout << std::endl;

  // Copy to host and verify equivalence.
  std::vector<ElementOutput> host_cutlass(ldc * N, 0);
  std::vector<ElementOutput> host_reference(ldc * N, 0);

  result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  //
  // Test for bit equivalence of results.
  //

  for(int i = 0; i < ldc * N; ++i) {
    if (host_cutlass[i] != host_reference[i]) {
      std::cerr << "Results differ at index " << i << " CUTLASS: " << static_cast<int>(host_cutlass[i]) << " Reference: " << static_cast<int>(host_reference[i]) << std::endl;

      // return cudaErrorUnknown;
    } else{
      // if(host_cutlass[i]!=0){
      //   std::cout << "Results are the same at index " << i << " CUTLASS: " << host_cutlass[i] << " Reference: " << host_reference[i] << std::endl;
      // }
    }
  }

  if (host_cutlass != host_reference) {
    std::cerr << "CUTLASS results incorrect." << std::endl;

    return cudaErrorUnknown;
  }

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *arg[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  // GEMM problem dimensions.
  int problem[3] = { 384, 384, 384 };

  for (int i = 1; i < argc && i < 4; ++i) {
    std::stringstream ss(arg[i]);
    ss >> problem[i - 1];
  }

  // Scalars used for linear scaling the result of the matrix product.
  data_type scalars[2] = { data_type(1), data_type(0) };

  for (int i = 4; i < argc && i < 6; ++i) {
    std::stringstream ss(arg[i]);
    ss >> scalars[i - 4];
  }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    problem[0],     // GEMM M dimension
    problem[1],     // GEMM N dimension
    problem[2],     // GEMM K dimension
    scalars[0],     // alpha
    scalars[1]      // beta
  );

  if (result == cudaSuccess) {
    std::cout << "Passed." << std::endl;
  }

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
