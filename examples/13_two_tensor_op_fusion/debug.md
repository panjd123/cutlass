# Iterations::kCount = Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous

static_assert(Iterations::kCount,
"Number of iterations must be non-zero");

Iterations::kCount = Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous

1. Detail::WarpAccessIterations::kContiguous = 2
2. Detail::kWarpsContiguous = 4

## Detail::WarpAccessIterations::kContiguous = ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous

Detail::WarpAccessIterations::kContiguous = ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous

3. ShapeInAccesses::kContiguous = 64
4. WarpThreadArrangement::kContiguous = 32

### ShapeInAccesses::kContiguous

ShapeInAccesses::kContiguous = Shape::kContiguous / kElementsPerAccess

5. Shape::kContiguous = 1024
6. kElementsPerAccess = 16

#### Shape::kContiguous

```
            instantiation of class "cutlass::whatIsN<N, condition, tag> [with N=1024, condition=1, tag=5]" at line 274 of /root/cutlass/include/cutlass/transform/pitch_linear_thread_map.h
            instantiation of class "cutlass::transform::PitchLinearWarpRakedThreadMap<Shape_, Threads, WarpThreadArrangement_, ElementsPerAccess>::Detail [with Shape_=cutlass::PitchLinearShape<1024, 1>, Threads=128, WarpThreadArrangement_=cutlass::PitchLinearShape<32, 1>, ElementsPerAccess=16]" at line 313 of /root/cutlass/include/cutlass/transform/pitch_linear_thread_map.h
            instantiation of class "cutlass::transform::PitchLinearWarpRakedThreadMap<Shape_, Threads, WarpThreadArrangement_, ElementsPerAccess> [with Shape_=cutlass::PitchLinearShape<1024, 1>, Threads=128, WarpThreadArrangement_=cutlass::PitchLinearShape<32, 1>, ElementsPerAccess=16]" at line 513 of /root/cutlass/include/cutlass/transform/pitch_linear_thread_map.h
            instantiation of class "cutlass::transform::TransposePitchLinearThreadMap<ThreadMap_, WarpThreadArrangement_> [with ThreadMap_=cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<1024, 1>, 128, cutlass::PitchLinearShape<32, 1>, 16>, WarpThreadArrangement_=cutlass::PitchLinearShape<2, 16>]" at line 51 of /root/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator.h
```

Shape::kContiguous = Shape::kM * kInterleavedK = 32 * 32

自顶向下 or 自底向上

```
1 2 3 4 5 ...
6 7 8 9 10 ...
...

InterleavedK = 2

1 6 2 7 3 8 4 9 5 10 ...
...
```

include/cutlass/gemm/threadblock/default_mma_core_sm80.h

Shape = ThreadBlockShape = {32, 384, 32}

#### kElementsPerAccess

static int const kElementsPerAccess =
    kAccessSizeInBits / sizeof_bits<ElementA>::value;

kAccessSizeInBits = 128

kElementsPerAccess = 128 / 8 = 16

cutlass/gemm/threadblock/default_mma_core_sm80.h:1867

### WarpThreadArrangement::kContiguous

cutlass/gemm/threadblock/default_mma_core_sm80.h:1901

32

## Detail::kWarpsContiguous

```
static int const kWarpsContiguous =
    (kWarpCount > WarpAccessIterations::kStrided
            ? kWarpCount / kWarpsStrided
            : 1);
```

7. kThreads = 128 = WarpkCount * 32;
8. kWarpSize = 32
9. kWarpCount = kThreads / kWarpSize = 4
10. WarpAccessIterations.<kContiguous, kStrided> = <2, 1>


include/cutlass/transform/threadblock/predicated_tile_access_iterator.h


```cpp
template <int N, int condition, int tag>
class whatIsN{
  static_assert(N == 100876 || !condition, "whatIsN");
};

template <typename T, int condition, int tag>
class whatIsT{
  static_assert(sizeof(T) == 100876 || !condition, "whatIsT");
};
```