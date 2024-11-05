#!/root/miniconda3/bin/python
import subprocess
import re
import pandas
import tqdm
import numpy as np
from parabuild import Parabuild
import os


shmem_path = "examples/13_two_tensor_op_fusion/fused_two_gemms_s8_sm80_shmem.cu"
lines = open(shmem_path, "r").readlines()
lines = [line.strip() for line in lines]
# #ifndef TESTN1
shmem_n_line = lines.index("#ifndef TESTN1")

#  #define NONFUSED_INSERT_POINT 0
non_fused_line = lines.index("#define NONFUSED_INSERT_POINT 0")
#  #define FUSED_INSERT_POINT 0
fused_line = lines.index("#define FUSED_INSERT_POINT 0")

def modify_line(file, line_no, content):
    with open(file, "r") as f:
        lines = f.readlines()
    lines[line_no] = content
    with open(file, "w") as f:
        f.writelines(lines)

def modify_shmem(n = 384,
                 thread_block_shape_m = None,
                 thread_block_shape_n = None,
                 thread_block_shape_k = None,
                 warp_shape_m = None,
                 warp_shape_n = None,
                 warp_shape_k = None,
                 fused = True,
                 ignore_other = False,
                 shmem_path = shmem_path,
                 **kwargs
                 ):
    if "thread_block_shape" in kwargs:
        thread_block_shape_m, thread_block_shape_n, thread_block_shape_k = kwargs["thread_block_shape"]
    if "warp_shape" in kwargs:
        warp_shape_m, warp_shape_n, warp_shape_k = kwargs["warp_shape"]
         
    modify_line(shmem_path, shmem_n_line, f"#define TESTN1 {n}\n")
    modify_line(shmem_path, shmem_n_line + 1, f"#define TESTN2 {n}\n")
    
    if not fused:
        modify_line(shmem_path, non_fused_line, f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {thread_block_shape_n}, {thread_block_shape_k}>;\n")
        modify_line(shmem_path, non_fused_line + 1, f"  using WarpShape0 = cutlass::gemm::GemmShape<{warp_shape_m}, {warp_shape_n}, {warp_shape_k}>;\n")
        modify_line(shmem_path, non_fused_line + 2, f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {thread_block_shape_n}, {thread_block_shape_k}>;\n")
        modify_line(shmem_path, non_fused_line + 3, f"  using WarpShape1 = cutlass::gemm::GemmShape<{warp_shape_m}, {warp_shape_n}, {warp_shape_k}>;\n")
    elif not ignore_other:
        modify_line(shmem_path, non_fused_line, f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{128}, {64}, {64}>;\n")
        modify_line(shmem_path, non_fused_line + 1, f"  using WarpShape0 = cutlass::gemm::GemmShape<{64}, {64}, {64}>;\n")
        modify_line(shmem_path, non_fused_line + 2, f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{128}, {128}, {64}>;\n")
        modify_line(shmem_path, non_fused_line + 3, f"  using WarpShape1 = cutlass::gemm::GemmShape<{64}, {64}, {64}>;\n")
    
    if fused:
        modify_line(shmem_path, fused_line, f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {n}, {thread_block_shape_k}>;\n")
        modify_line(shmem_path, fused_line + 1, f"  using WarpShape0 = cutlass::gemm::GemmShape<{warp_shape_m}, {n//4}, {warp_shape_k}>;\n")
        modify_line(shmem_path, fused_line + 2, f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{thread_block_shape_m}, {n}, {thread_block_shape_k}>;\n")
        modify_line(shmem_path, fused_line + 3, f"  using WarpShape1 = cutlass::gemm::GemmShape<{warp_shape_m}, {n//4}, {warp_shape_k}>;\n")
    elif not ignore_other:
        modify_line(shmem_path, fused_line, f"  using ThreadblockShape0 = cutlass::gemm::GemmShape<{64}, {n}, {64}>;\n")
        modify_line(shmem_path, fused_line + 1, f"  using WarpShape0 = cutlass::gemm::GemmShape<{64}, {n//4}, {64}>;\n")
        modify_line(shmem_path, fused_line + 2, f"  using ThreadblockShape1 = cutlass::gemm::GemmShape<{64}, {n}, {64}>;\n")
        modify_line(shmem_path, fused_line + 3, f"  using WarpShape1 = cutlass::gemm::GemmShape<{64}, {n//4}, {64}>;\n")

def run_shmem():
    try:
        output = subprocess.run(["cmake --build build --target 13_fused_two_gemms_s8_sm80_shmem \
                                && ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_shmem"],
                                shell=True, capture_output=True, check=True)
        output = output.stdout.decode("utf-8")
        shmem_non_fusion_time = re.findall(r"Non-fusion time (.*) ms",output)[0]
        shmem_fusion_time = re.findall(r"Fusion time (.*) ms", output)[0]
    except subprocess.CalledProcessError as e:
        shmem_non_fusion_time = "N/A"
        shmem_fusion_time = "N/A"
        output = e.stderr.decode("utf-8")
    return shmem_non_fusion_time, shmem_fusion_time, output

def task(workspace, n, thread_block_shape, warp_shape, fused):
    file_path = os.path.join(workspace, shmem_path)
    modify_shmem(shmem_path=file_path, n=n, thread_block_shape=thread_block_shape, warp_shape=warp_shape, fused=fused)
    try:
        output = subprocess.run(["cmake --build build --target 13_fused_two_gemms_s8_sm80_shmem \
                                && ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_shmem"],
                                shell=True, capture_output=True, check=True, cwd=workspace)
        output = output.stdout.decode("utf-8")
        shmem_non_fusion_time = re.findall(r"Non-fusion time (.*) ms",output)[0]
        shmem_fusion_time = re.findall(r"Fusion time (.*) ms", output)[0]
    except subprocess.CalledProcessError as e:
        shmem_non_fusion_time = "N/A"
        shmem_fusion_time = "N/A"
        output = e.stderr.decode("utf-8")
    return n, thread_block_shape, warp_shape, fused, shmem_non_fusion_time, shmem_fusion_time, None

def post_task_maker():
    data = []
    def post_task(x):
        if x[-2] == "N/A" or x[-3] == "N/A":
            return data
        data.append(x)
        return data
    return post_task

def main(debug=True, rg=False):
    pb = Parabuild(".",
                task,
                init_commands=[
                    ["rm", "-rf", "build"],
                    ["cmake", "-B", "build" ,"-DCUTLASS_NVCC_ARCHS=80" ,"-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=80", "-DCUTLASS_ENABLE_CUBLAS=ON" ,"-DCUTLASS_ENABLE_CUDNN=ON" ,"."]
                ],
                enable_tqdm=True,
                clean_workspace=True,
                excludes=[".git", "docs", "media", "build", "*.ncu-rep", "*.csv"],
                workspace_dir="../cutlass_workspace")
    
    n_space = [384]
    # search_space = [4,6,8,12,16,24,32,48,64,96,128]
    search_space = [64, 96]
    # search_space = [32, 64, 96, 128]
    # search_space = [64, 96, 128]
    bar = tqdm.tqdm(total=len(n_space)*len(search_space)**6 + len(n_space)*len(search_space)**4)
    for n in n_space:
        for fused in [True, False]:
            for thread_block_shape_m in search_space:
                for thread_block_shape_n in search_space if not fused else [n]:
                    for thread_block_shape_k in search_space:
                        for warp_shape_m in search_space:
                            for warp_shape_n in search_space if not fused else [n//4]:
                                for warp_shape_k in search_space:
                                    pb.add_task_kwargs(
                                        {
                                            "n": n,
                                            "thread_block_shape": (thread_block_shape_m, thread_block_shape_n, thread_block_shape_k),
                                            "warp_shape": (warp_shape_m, warp_shape_n, warp_shape_k),
                                            "fused": fused
                                        }
                                    )
    resutls = pb.join()
    print(resutls)

if __name__ == "__main__":
    main()