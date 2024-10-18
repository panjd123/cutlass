#!/root/miniconda3/envs/pytorch/bin/python
import subprocess
import re
import pandas
import tqdm
import numpy as np

rf_path = "examples/13_two_tensor_op_fusion/fused_two_gemms_s8_sm80_rf.cu"
rf_line = 50 - 1
shmem_path = "examples/13_two_tensor_op_fusion/fused_two_gemms_s8_sm80_shmem.cu"
shmem_line = 58 - 1

def modify_line(file, line_no, content):
    with open(file, "r") as f:
        lines = f.readlines()
    lines[line_no] = content
    with open(file, "w") as f:
        f.writelines(lines)

def main(debug=True, rg=False):
    data = []
    for n in tqdm.tqdm([128, 256, 384]): # 32 * np.arange(1, 17)
        modify_line(rf_path, rf_line, f"#define TESTN1 {n}\n")
        modify_line(rf_path, rf_line + 1, f"#define TESTN2 {n}\n")
        modify_line(shmem_path, shmem_line, f"#define TESTN1 {n}\n")
        modify_line(shmem_path, shmem_line + 1, f"#define TESTN2 {n}\n")
        try:
            if not rg:
                raise subprocess.CalledProcessError(1, "dummy")
            output = subprocess.run(["cmake --build build --target 13_fused_two_gemms_s8_sm80_rf \
                                     && ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_rf"],
                                    shell=True, capture_output=True, check=True)
            output = output.stdout.decode("utf-8")
            rf_non_fusion_time = re.findall(r"Non-fusion time (.*) ms",output)[0]
            rf_fusion_time = re.findall(r"Fusion time (.*) ms", output)[0]
            if debug:
                print(output)
        except subprocess.CalledProcessError as e:
            rf_non_fusion_time = "N/A"
            rf_fusion_time = "N/A"
        try:
            output = subprocess.run(["cmake --build build --target 13_fused_two_gemms_s8_sm80_shmem \
                                     && ./build/examples/13_two_tensor_op_fusion/13_fused_two_gemms_s8_sm80_shmem"],
                                    shell=True, capture_output=True, check=True)
            output = output.stdout.decode("utf-8")
            shmem_non_fusion_time = re.findall(r"Non-fusion time (.*) ms",output)[0]
            shmem_fusion_time = re.findall(r"Fusion time (.*) ms", output)[0]
            if debug:
                print(output)
        except subprocess.CalledProcessError as e:
            shmem_non_fusion_time = "N/A"
            shmem_fusion_time = "N/A"
        tqdm.tqdm.write(f"{n}, {n}, {rf_non_fusion_time}, {rf_fusion_time}, {shmem_non_fusion_time}, {shmem_fusion_time}")
        data.append([n, n, rf_non_fusion_time, rf_fusion_time, shmem_non_fusion_time, shmem_fusion_time])
    df = pandas.DataFrame(data, columns=["N1", "N2", "RF non-fusion time", "RF fusion time", "Shmem non-fusion time", "Shmem fusion time"])
    # df.to_csv("fused_gemm.csv", index=False)

if __name__ == "__main__":
    main()