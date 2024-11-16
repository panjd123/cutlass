#!/root/miniconda3/bin/python
import subprocess
import re
import pandas
import tqdm
import numpy as np


shmem_path = "examples/13_two_tensor_op_fusion/fused_two_gemms_s4_sm80_shmem.cu"
lines = open(shmem_path, "r").readlines()
lines = [line.strip() for line in lines]
# #ifndef TESTN1
shmem_n1_line = lines.index("#ifndef TESTN1") + 1
shmem_n2_line = lines.index("#ifndef TESTN2") + 1

#  #define NONFUSED_INSERT_POINT 0
non_fused_line = lines.index("#define NONFUSED_INSERT_POINT 0") + 1
#  #define FUSED_INSERT_POINT 0
fused_line = lines.index("#define FUSED_INSERT_POINT 0") + 1


def modify_line(file, line_no, content):
    with open(file, "r") as f:
        lines = f.readlines()
    lines[line_no] = content
    with open(file, "w") as f:
        f.writelines(lines)


def modify_shmem(
    n,
    thread_block_shape_m=None,
    thread_block_shape_n=None,
    thread_block_shape_k=None,
    warp_shape_m=None,
    warp_shape_n=None,
    warp_shape_k=None,
    fused=True,
    ignore_other=False,
    **kwargs,
):
    if "thread_block_shape" in kwargs:
        thread_block_shape_m, thread_block_shape_n, thread_block_shape_k = kwargs[
            "thread_block_shape"
        ]
    if "warp_shape" in kwargs:
        warp_shape_m, warp_shape_n, warp_shape_k = kwargs["warp_shape"]

    modify_line(shmem_path, shmem_n1_line, f"#define TESTN1 {n}\n")
    modify_line(shmem_path, shmem_n2_line, f"#define TESTN2 {n}\n")

    if not fused:
        modify_line(
            shmem_path,
            non_fused_line,
            f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {thread_block_shape_n}, {thread_block_shape_k}>;\n",
        )
        modify_line(
            shmem_path,
            non_fused_line + 1,
            f"  using WarpShape0 = cutlass::gemm::GemmShape<{warp_shape_m}, {warp_shape_n}, {warp_shape_k}>;\n",
        )
        modify_line(
            shmem_path,
            non_fused_line + 2,
            f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {thread_block_shape_n}, {thread_block_shape_k}>;\n",
        )
        modify_line(
            shmem_path,
            non_fused_line + 3,
            f"  using WarpShape1 = cutlass::gemm::GemmShape<{warp_shape_m}, {warp_shape_n}, {warp_shape_k}>;\n",
        )
    elif not ignore_other:
        modify_line(
            shmem_path,
            non_fused_line,
            f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{128}, {96}, {128}>;\n",
        )
        modify_line(
            shmem_path,
            non_fused_line + 1,
            f"  using WarpShape0 = cutlass::gemm::GemmShape<{32}, {96}, {128}>;\n",
        )
        modify_line(
            shmem_path,
            non_fused_line + 2,
            f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{128}, {96}, {128}>;\n",
        )
        modify_line(
            shmem_path,
            non_fused_line + 3,
            f"  using WarpShape1 = cutlass::gemm::GemmShape<{32}, {96}, {128}>;\n",
        )

    if fused:
        modify_line(
            shmem_path,
            fused_line,
            f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {n}, {thread_block_shape_k}>;\n",
        )
        modify_line(
            shmem_path,
            fused_line + 1,
            f"  using WarpShape0 = cutlass::gemm::GemmShape<{warp_shape_m}, {n//4}, {warp_shape_k}>;\n",
        )
        modify_line(
            shmem_path,
            fused_line + 2,
            f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {n}, {thread_block_shape_k}>;\n",
        )
        modify_line(
            shmem_path,
            fused_line + 3,
            f"  using WarpShape1 = cutlass::gemm::GemmShape<{warp_shape_m}, {n//4}, {warp_shape_k}>;\n",
        )
    elif not ignore_other:
        modify_line(
            shmem_path,
            fused_line,
            f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{32}, {n}, {128}>;\n",
        )
        modify_line(
            shmem_path,
            fused_line + 1,
            f"  using WarpShape0 = cutlass::gemm::GemmShape<{32}, {n//4}, {128}>;\n",
        )
        modify_line(
            shmem_path,
            fused_line + 2,
            f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{32}, {n}, {128}>;\n",
        )
        modify_line(
            shmem_path,
            fused_line + 3,
            f"  using WarpShape1 = cutlass::gemm::GemmShape<{32}, {n//4}, {128}>;\n",
        )


def run_shmem():
    try:
        output = subprocess.run(
            [
                "cmake --build build --target 13_fused_two_gemms_s4_sm80_shmem \
                                && ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s4_sm80_shmem"
            ],
            shell=True,
            capture_output=True,
            check=False,
        )
        output = output.stdout.decode("utf-8")
        shmem_non_fusion_time = re.findall(r"Non-fusion time (.*) ms", output)[0]
        shmem_fusion_time = re.findall(r"Fusion time (.*) ms", output)[0]
    except subprocess.CalledProcessError as e:
        shmem_non_fusion_time = "N/A"
        shmem_fusion_time = "N/A"
        output = e.stderr.decode("utf-8")
    except IndexError:
        shmem_non_fusion_time = "N/A"
        shmem_fusion_time = "N/A"
    return shmem_non_fusion_time, shmem_fusion_time, output


def main(debug=True, rg=False):
    data = []
    try:
        n_space = [384]
        # search_space = [4,6,8,12,16,24,32,48,64,96,128]
        search_space = [32, 48, 64, 96, 128]
        # search_space = [64, 96, 128]
        datas = []
        for n in n_space:
            for fused in [False, True]:
                for thread_block_shape_m in search_space:
                    for thread_block_shape_n in search_space if not fused else [n]:
                        for thread_block_shape_k in [
                            k for k in search_space if k >= 128
                        ]:
                            for warp_shape_m in search_space:
                                for warp_shape_n in (
                                    search_space if not fused else [n // 4]
                                ):
                                    for warp_shape_k in [
                                        k for k in search_space if k >= 128
                                    ]:
                                        datas.append(
                                            [
                                                n,
                                                thread_block_shape_m,
                                                thread_block_shape_n,
                                                thread_block_shape_k,
                                                warp_shape_m,
                                                warp_shape_n,
                                                warp_shape_k,
                                                fused,
                                            ]
                                        )

        datas = [
            [384, 32, 384, 128, 32, 96, 128, True],
            [384, 128, 96, 128, 32, 96, 128, False],
        ]

        bar = tqdm.tqdm(total=len(datas))
        for (
            n,
            thread_block_shape_m,
            thread_block_shape_n,
            thread_block_shape_k,
            warp_shape_m,
            warp_shape_n,
            warp_shape_k,
            fused,
        ) in datas:
            modify_shmem(
                n,
                thread_block_shape_m,
                thread_block_shape_n,
                thread_block_shape_k,
                warp_shape_m,
                warp_shape_n,
                warp_shape_k,
                fused,
            )

            shmem_non_fusion_time, shmem_fusion_time, output = run_shmem()
            if debug:
                print(output)

            if shmem_non_fusion_time != "N/A" and shmem_fusion_time != "N/A":
                data.append(
                    [
                        n,
                        n,
                        thread_block_shape_m,
                        thread_block_shape_n,
                        thread_block_shape_k,
                        warp_shape_m,
                        warp_shape_n,
                        warp_shape_k,
                        fused,
                        shmem_non_fusion_time,
                        shmem_fusion_time,
                    ]
                )
                tqdm.tqdm.write(
                    f"{fused}, n={n}, thread_block_shape=({thread_block_shape_m}, {thread_block_shape_n}, {thread_block_shape_k}), warp_shape=({warp_shape_m}, {warp_shape_n}, {warp_shape_k}), shmem_non_fusion_time={shmem_non_fusion_time}, shmem_fusion_time={shmem_fusion_time}"
                )
            bar.update(1)
    except KeyboardInterrupt:
        pass
    df = pandas.DataFrame(
        data,
        columns=[
            "N1",
            "N2",
            "ThreadBlockShapeM",
            "ThreadBlockShapeN",
            "ThreadBlockShapeK",
            "WarpShapeM",
            "WarpShapeN",
            "WarpShapeK",
            "Fused",
            "ShmemNonFusionTime",
            "ShmemFusionTime",
        ],
    )
    df.to_csv("fused_gemm.csv", index=False)


if __name__ == "__main__":
    main()
    # modify_shmem(fused=False, ignore_other=True, n=384, thread_block_shape=(128, 96, 64), warp_shape=(32, 96, 64))
    # shmem_non_fusion_time, _, _ = run_shmem()
    # print(shmem_non_fusion_time)
    # modify_shmem(fused=True, ignore_other=True, n=384, thread_block_shape=(32, 384, 64), warp_shape=(32, 96, 64))
    # _, shmem_fusion_time, _ = run_shmem()
    # print(shmem_fusion_time)
