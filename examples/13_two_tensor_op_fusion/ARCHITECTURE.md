# fused_gemm_shmem

key codes:

- examples/13_two_tensor_op_fusion/kernel/b2b_gemm.h:547
- examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage_smem_accumulator.h:431

## kernel launch

- examples/13_two_tensor_op_fusion/device/b2b_gemm.h:327

## kernel

- examples/13_two_tensor_op_fusion/kernel/b2b_gemm.h:547

```cpp
// examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm_smem_accumulator.h:375
using B2bGemmKernel = kernel::B2bGemm<B2bMma, Epilogue, ThreadblockSwizzle>;
//                            ^examples/13_two_tensor_op_fusion/kernel/b2b_gemm.h:104
```

```cpp
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;
    run_with_swizzle(params, shared_storage, threadblock_swizzle);
  }

  /// Executes one GEMM with an externally-provided swizzling function
  CUTLASS_DEVICE
  void run_with_swizzle(Params const &params, SharedStorage &shared_storage, ThreadblockSwizzle& threadblock_swizzle) {

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    ElementA0 *ptr_A0 = static_cast<ElementA0 *>(params.ref_A0.data());
    ElementB0 *ptr_B0 = static_cast<ElementB0 *>(params.ref_B0.data());
    ElementB1 *ptr_B1 = static_cast<ElementB1 *>(params.ref_B1.data());

    ScaleBiasData *ptr_Bias0 = static_cast<ScaleBiasData *>(params.ref_Bias0.data());
    ScaleBiasData *ptr_Scale0 = static_cast<ScaleBiasData *>(params.ref_Scale0.data());

    int offset_k_0 = 0;
    int offset_k_1 = 0;

    int problem_size_k_0 = params.problem_size_0.k();
    int problem_size_k_1 = params.problem_size_1.k();

    if (params.mode == GemmUniversalMode::kGemm) {

      // Problem size is a function of threadblock index in the K dimension
      problem_size_k_0 = min(
        problem_size_k_0,
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size_0);

      // Problem size is a function of threadblock index in the K dimension
      problem_size_k_1 = min(
        problem_size_k_1,
        (threadblock_tile_offset.k() + 1) * params.gemm_k_size_1);

      offset_k_0 = threadblock_tile_offset.k() * params.gemm_k_size_0;
      offset_k_1 = threadblock_tile_offset.k() * params.gemm_k_size_1;
    }

    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_A0 += threadblock_tile_offset.k() * params.batch_stride_A0;
      ptr_B0 += threadblock_tile_offset.k() * params.batch_stride_B0;
      ptr_B1 += threadblock_tile_offset.k() * params.batch_stride_B1;
      ptr_Bias0 += threadblock_tile_offset.k() * params.batch_stride_Bias0;
      ptr_Scale0 += threadblock_tile_offset.k() * params.batch_stride_Scale0;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A0{
      threadblock_tile_offset.m() * B2bMma::Shape0::kM,
      offset_k_0,
    };

    cutlass::MatrixCoord tb_offset_B0{
      offset_k_0,
      threadblock_tile_offset.n() * B2bMma::Shape0::kN
    };

    cutlass::MatrixCoord tb_offset_B1{
      offset_k_1,
      threadblock_tile_offset.n() * B2bMma::Shape1::kN
    };

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations_0 = (problem_size_k_0 - tb_offset_A0.column() + B2bMma::Shape0::kK - 1) / B2bMma::Shape0::kK;

    // Compute threadblock-scoped matrix multiply-add
    // int gemm_k_iterations_1 = (problem_size_k_1 - tb_offset_B1.row() + B2bMma::Shape1::kK - 1) / B2bMma::Shape1::kK;


    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename B2bMma::IteratorA0 iterator_A0(
      params.params_A0,
      ptr_A0,
      {params.problem_size_0.m(), problem_size_k_0},
      thread_idx,
      tb_offset_A0);

    typename B2bMma::IteratorB0 iterator_B0(
      params.params_B0,
      ptr_B0,
      {problem_size_k_0, params.problem_size_0.n()},
      thread_idx,
      tb_offset_B0);

    typename B2bMma::IteratorB1 iterator_B1(
      params.params_B1,
      ptr_B1,
      {problem_size_k_1, params.problem_size_1.n()},
      thread_idx,
      tb_offset_B1);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    // Construct iterators to accumulator scale/bias vector
    typename B2bMma::IteratorAccumulatorScaleBias iterator_Scale0(
      ptr_Scale0,
      {1, params.problem_size_0.n()},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_offset.n() * B2bMma::Shape0::kN
      )
    );

    typename B2bMma::IteratorAccumulatorScaleBias iterator_Bias0(
      ptr_Bias0,
      {1, params.problem_size_0.n()},
      thread_idx,
      warp_idx,
      MatrixCoord(
        0, threadblock_tile_offset.n() * B2bMma::Shape0::kN
      )
    );

    //
    // Main loop
    //

    OutputOp0 output_op_0(params.output_op_0);

    if (cutlass::gemm::threadblock::detail::IsGroupedSwizzle<ThreadblockSwizzle>::value) {
      // Wait for all threads to finish their epilogue phases from the previous tile.
      __syncthreads();
    }

    // Construct thread-scoped matrix multiply
    B2bMma b2bMma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx, params.problem_size_0.n());

    typename B2bMma::FragmentC0 src_accum;
    typename B2bMma::FragmentC1 accumulators;

    src_accum.clear();
    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    b2bMma(gemm_k_iterations_0, accumulators, iterator_A0, iterator_B0,
      iterator_Scale0, iterator_Bias0, iterator_B1, src_accum, output_op_0);

    //
    // Epilogue
    //

    OutputOp1 output_op_1(params.output_op_1);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * B2bMma::Shape1::kM,
      threadblock_tile_offset.n() * B2bMma::Shape1::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    ElementC *ptr_C1 = static_cast<ElementC *>(params.ref_C1.data());
    ElementC *ptr_D1 = static_cast<ElementC *>(params.ref_D1.data());

    // Construct the semaphore.
    Semaphore semaphore(params.semaphore + block_idx, thread_idx);

    if (params.mode == GemmUniversalMode::kGemm) {
      // If performing a reduction via split-K, fetch the initial synchronization

      if (params.grid_tiled_shape.k() > 1) {
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        output_op_1.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
      }
    }
    else if (params.mode == GemmUniversalMode::kBatched) {
      ptr_C1 += threadblock_tile_offset.k() * params.batch_stride_C1;
      ptr_D1 += threadblock_tile_offset.k() * params.batch_stride_D1;
    }

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C1(
      params.params_C1,
      ptr_C1,
      params.problem_size_1.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D1(
      params.params_D1,
      ptr_D1,
      params.problem_size_1.mn(),
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {

      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_offset.k()) {
        iterator_C1 = iterator_D1;
      }

      semaphore.wait(threadblock_tile_offset.k());

      __threadfence();
    }

    // Execute the epilogue operator to update the destination tensor.
    epilogue(output_op_1, iterator_D1, accumulators, iterator_C1);

    //
    // Release the semaphore
    //

    if (params.mode == GemmUniversalMode::kGemm && params.grid_tiled_shape.k() > 1) {

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_offset.k() + 1;
      }

      __threadfence();
      semaphore.release(lock);
    }
  }
```

## b2bMma

- examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage_smem_accumulator.h:431

```cpp
// examples/13_two_tensor_op_fusion/kernel/default_b2b_gemm_smem_accumulator.h:291
using B2bMma = typename cutlass::gemm::threadblock::DefaultB2bMma<..., true, true>::ThreadblockB2bMma;
//                                                  ^examples/13_two_tensor_op_fusion/threadblock/default_b2b_mma_smem_accumulator.h:503

using ThreadblockB2bMma = cutlass::gemm::threadblock::B2bMmaMultistageSmemAccumulator<...>;
//                                                    ^examples/13_two_tensor_op_fusion/threadblock/b2b_mma_multistage_smem_accumulator.h:115
```

```cpp
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations_0,
      ///< destination accumulator tile
      FragmentC1 &accum,
      ///< iterator over A0 operand in global memory
      IteratorA0 iterator_A0,
      ///< iterator over B0 operand in global memory
      IteratorB0 iterator_B0,
      ///< iterator over A1 operand scale vector in global memory
      IteratorAccumulatorScaleBias iterator_accum0_scale,
      ///< iterator over A1 operand bias vector in global memory
      IteratorAccumulatorScaleBias iterator_accum0_bias,
      ///< iterator over B1 operand in global memory
      IteratorB1 iterator_B1,
      ///< initial value of accumulator
      FragmentC0 const &src_accum,
      ///< epilogue operation after 1st Gemm
      OutputOp output_op_0)
    {
    //
    // Prologue
    //

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations_0) {

      iterator_A0.clear_mask(gemm_k_iterations_0 == 0);
      iterator_B0.clear_mask(gemm_k_iterations_0 == 0);

      iterator_A0.set_iteration_index(0);
      this->smem_iterator_A0_.set_iteration_index(0);

      // cp.async for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsA0; ++j) {
        typename IteratorA0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA0::AccessType *>(
                this->smem_iterator_A0_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA0::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA0::Element>::value *
              IteratorA0::ThreadMap::kElementsPerAccess /
              IteratorA0::kAccessesPerVector / 8;

          int src_bytes = (iterator_A0.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA0>(
              dst_ptr + v, iterator_A0.get(), iterator_A0.valid());

          ++iterator_A0;
        }

        ++this->smem_iterator_A0_;
      }

      iterator_B0.set_iteration_index(0);
      this->smem_iterator_B0_.set_iteration_index(0);

      // cp.async for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsB0; ++j) {
        typename IteratorB0::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB0::AccessType *>(
                this->smem_iterator_B0_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB0::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB0::Element>::value *
              IteratorB0::ThreadMap::kElementsPerAccess /
              IteratorB0::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB0>(
              dst_ptr + v, iterator_B0.get(), iterator_B0.valid());

          ++iterator_B0;
        }

        ++this->smem_iterator_B0_;
      }

      // Move to the next stage
      iterator_A0.add_tile_offset({0, 1});
      iterator_B0.add_tile_offset({1, 0});

      this->smem_iterator_A0_.add_tile_offset({0, 1});
      this->smem_iterator_B0_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    FragmentC0 accum0 = src_accum;

    // DEPBAR+SYNC
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA0 warp_loaded_frag_A0[2];
    WarpLoadedFragmentB0 warp_loaded_frag_B0[2];
    WarpTransformedFragmentA0 warp_transformed_frag_A0[2];
    WarpTransformedFragmentB0 warp_transformed_frag_B0[2];

    Operator0 warp_mma0;

    this->warp_tile_iterator_A0_.set_kgroup_index(0);
    this->warp_tile_iterator_B0_.set_kgroup_index(0);

    this->warp_tile_iterator_A0_.load(warp_loaded_frag_A0[0]);
    this->warp_tile_iterator_B0_.load(warp_loaded_frag_B0[0]);

    ++this->warp_tile_iterator_A0_;
    ++this->warp_tile_iterator_B0_;

    iterator_A0.clear_mask(gemm_k_iterations_0 == 0);
    iterator_B0.clear_mask(gemm_k_iterations_0 == 0);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma0.transform(warp_transformed_frag_A0[0], warp_transformed_frag_B0[0],
                       warp_loaded_frag_A0[0], warp_loaded_frag_B0[0]);

    //
    // Mainloop
    //

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations_0 > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations0;
           ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);
        this->warp_tile_iterator_B0_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations0);
        
        this->warp_tile_iterator_A0_.load(warp_loaded_frag_A0[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B0_.load(warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A0_;
        ++this->warp_tile_iterator_B0_;

        if (warp_mma_k > 0)
          warp_mma0.transform(warp_transformed_frag_A0[warp_mma_k % 2],
                             warp_transformed_frag_B0[warp_mma_k % 2],
                             warp_loaded_frag_A0[warp_mma_k % 2],
                             warp_loaded_frag_B0[warp_mma_k % 2]);

        warp_mma0(
          accum0, 
          warp_transformed_frag_A0[warp_mma_k % 2],
          warp_transformed_frag_B0[warp_mma_k % 2], 
          accum0
        );

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations0 - 1) {
          int group_start_iteration_A0, group_start_iteration_B0;

          group_start_iteration_A0 = warp_mma_k * Detail::kAccessesPerGroupA0;
          group_start_iteration_B0 = warp_mma_k * Detail::kAccessesPerGroupB0;

          copy_tiles_and_advance_0(iterator_A0, iterator_B0, group_start_iteration_A0, 
                               group_start_iteration_B0);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations0) {
          int group_start_iteration_A0, group_start_iteration_B0;
          group_start_iteration_A0 =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA0;
          group_start_iteration_B0 =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB0;

          copy_tiles_and_advance_0(iterator_A0, iterator_B0, group_start_iteration_A0, 
                               group_start_iteration_B0);

          // Inserts a memory fence between stages of cp.async instructions.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages have committed.
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A0.add_tile_offset({0, 1});
          iterator_B0.add_tile_offset({1, 0});

          this->smem_iterator_A0_.add_tile_offset({0, 1});
          this->smem_iterator_B0_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A0_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B0_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A0_.add_tile_offset(
                {0, -Base::kStages * Policy0::kPartitionsK *
                        Base::kWarpGemmIterations0});
            this->warp_tile_iterator_B0_.add_tile_offset(
                {-Base::kStages * Policy0::kPartitionsK *
                     Base::kWarpGemmIterations0,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations_0;
          iterator_A0.clear_mask(gemm_k_iterations_0 == 0);
          iterator_B0.clear_mask(gemm_k_iterations_0 == 0);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations0)
          warp_mma0.transform(warp_transformed_frag_A0[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B0[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A0[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B0[(warp_mma_k + 1) % 2]);
      }

    }

    // Insert fence and wait for all outstanding cp.async operations to commit.
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    /// Epilogue for the first Implicit Gemm
    Epilogue0 epilogue0;

    epilogue0(output_op_0, smem_iterator_D0_, accum0, iterator_accum0_scale, iterator_accum0_bias);

    __syncthreads();


    // 2nd Gemm

    //
    // Prologue
    //
    int gemm_k_iterations_1 = Shape0::kN / Shape1::kK;

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations_1) {

      iterator_B1.clear_mask(gemm_k_iterations_1 == 0);

      iterator_B1.set_iteration_index(0);
      this->smem_iterator_B1_.set_iteration_index(0);

      // cp.async for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::TBLoadIterationsB1; ++j) {
        typename IteratorB1::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB1::AccessType *>(
                this->smem_iterator_B1_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB1::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB1::Element>::value *
              IteratorB1::ThreadMap::kElementsPerAccess /
              IteratorB1::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB1>(
              dst_ptr + v, iterator_B1.get(), iterator_B1.valid());

          ++iterator_B1;
        }

        ++this->smem_iterator_B1_;
      }

      // Move to the next stage
      iterator_B1.add_tile_offset({1, 0});

      this->smem_iterator_B1_.add_tile_offset({1, 0});

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // DEPBAR+SYNC
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA1 warp_loaded_frag_A1[2];
    WarpLoadedFragmentB1 warp_loaded_frag_B1[2];
    WarpTransformedFragmentA1 warp_transformed_frag_A1[2];
    WarpTransformedFragmentB1 warp_transformed_frag_B1[2];

    Operator1 warp_mma1;

    warp_tile_iterator_A1_.load(warp_loaded_frag_A1[0]);
    ++warp_tile_iterator_A1_;

    this->warp_tile_iterator_B1_.set_kgroup_index(0);
    this->warp_tile_iterator_B1_.load(warp_loaded_frag_B1[0]);
    ++this->warp_tile_iterator_B1_;

    iterator_B1.clear_mask(gemm_k_iterations_1 == 0);

    smem_write_stage_idx = Base::kStages - 1;
    smem_read_stage_idx = 0;

    warp_mma1.transform(warp_transformed_frag_A1[0], warp_transformed_frag_B1[0],
                       warp_loaded_frag_A1[0], warp_loaded_frag_B1[0]);

    //
    // Mainloop
    //

    CUTLASS_PRAGMA_UNROLL
    for ( gemm_k_iterations_1 = Shape0::kN / Shape1::kK - (Base::kStages - 1); 
            gemm_k_iterations_1 > (-Base::kStages + 1); gemm_k_iterations_1--) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations1;
           ++warp_mma_k) {

        // Load warp-level tile from accumulator fragment
        // skip warp tile loading for the last kgroup
        if(gemm_k_iterations_1 > (-Base::kStages + 2) || warp_mma_k < Base::kWarpGemmIterations1 - 1) {
            warp_tile_iterator_A1_.load(warp_loaded_frag_A1[(warp_mma_k + 1) % 2]);
        }
        ++warp_tile_iterator_A1_;

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.
        this->warp_tile_iterator_B1_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations1);
        this->warp_tile_iterator_B1_.load(warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
        ++this->warp_tile_iterator_B1_;


        if (warp_mma_k > 0)
          warp_mma1.transform(warp_transformed_frag_A1[warp_mma_k % 2],
                             warp_transformed_frag_B1[warp_mma_k % 2],
                             warp_loaded_frag_A1[warp_mma_k % 2],
                             warp_loaded_frag_B1[warp_mma_k % 2]);


        warp_mma1(
          accum, 
          warp_transformed_frag_A1[warp_mma_k % 2],
          warp_transformed_frag_B1[warp_mma_k % 2], 
          accum
        );

        // Issue global->shared copies for the this stage
        if (warp_mma_k < Base::kWarpGemmIterations1 - 1) {
          int group_start_iteration_B1;

          group_start_iteration_B1 = warp_mma_k * Detail::kAccessesPerGroupB1;

          copy_tiles_and_advance_1(iterator_B1, group_start_iteration_B1);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations1) {
          int group_start_iteration_B1;
          group_start_iteration_B1 =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB1;

          copy_tiles_and_advance_1(iterator_B1, group_start_iteration_B1);

          // Inserts a memory fence between stages of cp.async instructions.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages have committed.
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_B1.add_tile_offset({1, 0});

          this->smem_iterator_B1_.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_B1_.add_tile_offset(
                {-Base::kStages * Policy1::kPartitionsK *
                     Base::kWarpGemmIterations1,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          iterator_B1.clear_mask(gemm_k_iterations_1 == 1);
        }

        // Do any conversions feeding the first stage at the end of the loop so
        // we can start right away on mma instructions
        if (warp_mma_k + 1 == Base::kWarpGemmIterations1)
          warp_mma1.transform(warp_transformed_frag_A1[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B1[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_A1[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_B1[(warp_mma_k + 1) % 2]);
      }

    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }
```
