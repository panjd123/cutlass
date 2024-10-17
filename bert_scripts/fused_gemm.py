import subprocess
import re
import pandas
import tqdm

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

def main():
    data = []
    for n in tqdm.tqdm([128, 256, 512, 1024, 2048, 4096]): # , 128, 256, 512, 1024, 2048, 4096
        modify_line(rf_path, rf_line, f"#define TESTN1 {n}\n")
        modify_line(rf_path, rf_line + 1, f"#define TESTN2 {n}\n")
        modify_line(shmem_path, shmem_line, f"#define TESTN1 {n}\n")
        modify_line(shmem_path, shmem_line + 1, f"#define TESTN2 {n}\n")
        output = subprocess.run(["./bert_scripts/fused_gemm.sh"], shell=True, capture_output=True)
        output = output.stdout.decode("utf-8")
        lines = output.split("Device")
        rf_non_fusion_time = re.findall(r"Non-fusion time (.*) ms", lines[1])[0]
        rf_fusion_time = re.findall(r"Fusion time (.*) ms", lines[1])[0]
        shmem_non_fusion_time = re.findall(r"Non-fusion time (.*) ms", lines[2])[0]
        shmem_fusion_time = re.findall(r"Fusion time (.*) ms", lines[2])[0]
        tqdm.tqdm.write(f"{n}, {n}, {rf_non_fusion_time}, {rf_fusion_time}, {shmem_non_fusion_time}, {shmem_fusion_time}")
        data.append([n, n, rf_non_fusion_time, rf_fusion_time, shmem_non_fusion_time, shmem_fusion_time])
    df = pandas.DataFrame(data, columns=["N1", "N2", "RF non-fusion time", "RF fusion time", "Shmem non-fusion time", "Shmem fusion time"])
    df.to_csv("fused_gemm.csv", index=False)
    
main()