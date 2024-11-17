import json
import re
import pandas as pd

def get_data(datas):
    output = []
    for result in datas:
        if result["status"] == 0:
            fused = "FThreadblockShapeK" in result["data"]
            if fused:
                ThreadblockShape = (result["data"]["FThreadblockShapeM"], result["data"]["FThreadblockShapeN"], result["data"]["FThreadblockShapeK"])
                WarpShape = (result["data"]["FWarpShapeM"], result["data"]["FWarpShapeN"], result["data"]["FWarpShapeK"])
                time = re.findall(r"Fusion time (.*) ms", result["stdout"])[0]
            else:
                ThreadblockShape = (result["data"]["NFThreadblockShapeM"], result["data"]["NFThreadblockShapeN"], result["data"]["NFThreadblockShapeK"])
                WarpShape = (result["data"]["NFWarpShapeM"], result["data"]["NFWarpShapeN"], result["data"]["NFWarpShapeK"])
                time = re.findall(r"Non-fusion time (.*) ms", result["stdout"])[0]
            output.append({"ThreadblockShape": ThreadblockShape, "WarpShape": WarpShape, "Fused": fused, "time": time})
    return output
    
if __name__ == "__main__":
    datas = json.load(open("example.json", "r"))
    output = get_data(datas)
    # print(output)
    df = pd.DataFrame(output)
    df.sort_values(by=["time"], inplace=True)
    df_fused = df[df["Fused"] == True].reset_index(drop=True)
    df_non_fused = df[df["Fused"] == False].reset_index(drop=True)
    print(df_fused)
    print(df_non_fused)