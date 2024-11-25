import json
import re
import pandas as pd
import argparse

def get_data(datas):
    output = []
    for result in datas:
        fused = result["data"]["Fused"]
        try:
            if fused:
                ThreadblockShape = (result["data"]["FThreadblockShapeM"], result["data"]["FThreadblockShapeN"], result["data"]["FThreadblockShapeK"])
                WarpShape = (result["data"]["FWarpShapeM"], result["data"]["FWarpShapeN"], result["data"]["FWarpShapeK"])
                time = re.findall(r"Fusion time (.*) ms", result["stdout"])[0]
            else:
                ThreadblockShape = (result["data"]["NFThreadblockShapeM"], result["data"]["NFThreadblockShapeN"], result["data"]["NFThreadblockShapeK"])
                WarpShape = (result["data"]["NFWarpShapeM"], result["data"]["NFWarpShapeN"], result["data"]["NFWarpShapeK"])
                time = re.findall(r"Non-fusion time (.*) ms", result["stdout"])[0]
            output.append({"ThreadblockShape": ThreadblockShape, "WarpShape": WarpShape, "Fused": fused, "time": time, "status": result["status"]})
        except IndexError:
            pass
    return output
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json", help="input json file", default="output.json")
    args = parser.parse_args()
    datas = json.load(open(args.json, "r"))
    output = get_data(datas)
    
    df = pd.DataFrame(output)
    df.sort_values(by=["time"], inplace=True)
    df_fused = df[df["Fused"] == True].reset_index(drop=True)
    df_non_fused = df[df["Fused"] == False].reset_index(drop=True)
    print(df_fused)
    print(df_non_fused)
    
    print("---------------------------------------")
    
    print(df_fused[df_fused["status"] == 0])
    print(df_non_fused[df_non_fused["status"] == 0])
    
    print("---------------------------------------")
    
    print(df_fused[df_fused["status"] >= 0])
    print(df_non_fused[df_non_fused["status"] >= 0])
    
    # df["fused"]["baseline"] 
    times = []
    for i, row in df_fused.iterrows():
        insert = False
        for j, row_non_fused in df_non_fused.iterrows():
            if row["ThreadblockShape"] == row_non_fused["ThreadblockShape"] and row["WarpShape"] == row_non_fused["WarpShape"]:
                times.append(row_non_fused["time"])
                insert = True
                break
            else:
                pass
                # print(row["ThreadblockShape"], row_non_fused["ThreadblockShape"], row["WarpShape"], row_non_fused["WarpShape"])
        if not insert:
            times.append(None)
    print(times)
    df_fused["baseline"] = times
    print(df_fused)