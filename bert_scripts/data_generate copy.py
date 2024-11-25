import json
import numpy as np

# {
# "N1": 384,
# "N2": 384,
# "FThreadblockShapeM" : 32,
# "FThreadblockShapeN" : 384,
# "FThreadblockShapeK" : 64,
# "FWarpShapeM": 32,
# "FWarpShapeN": 96,
# "FWarpShapeK": 64,
# "NFThreadblockShapeM" : 128,
# "NFThreadblockShapeN" : 96,
# "NFThreadblockShapeK" : 64,
# "NFWarpShapeM": 32,
# "NFWarpShapeN": 96,
# "NFWarpShapeK": 64
# }


def generate_json(fused=True):
    search_space = [32, 48, 64, 96, 128, 192]
    N = 384
    datas = []
    if not fused:
        for tbm in search_space:
            for tbn in search_space:
                for tbk in search_space:
                    for wsm in search_space:
                        if tbm % wsm != 0:
                            continue
                        for wsn in search_space:
                            if tbn % wsn != 0:
                                continue
                            for wsk in search_space:
                                if tbk % wsk != 0:
                                    continue
                                data = {
                                    "N1": N,
                                    "N2": N,
                                    "NFThreadblockShapeM": tbm,
                                    "NFThreadblockShapeN": tbn,
                                    "NFThreadblockShapeK": tbk,
                                    "NFWarpShapeM": wsm,
                                    "NFWarpShapeN": wsn,
                                    "NFWarpShapeK": wsk,
                                    "Fused": False
                                }
                                datas.append(data)
    else:
        for tbm in search_space:
            for tbn in [N]:
                for tbk in search_space:
                    for wsm in search_space:
                        if tbm % wsm != 0:
                            continue
                        for wsn in search_space:
                            if tbn % wsn != 0:
                                continue
                            for wsk in search_space:
                                if tbk % wsk != 0:
                                    continue
                                data = {
                                    "N1": N,
                                    "N2": N,
                                    "FThreadblockShapeM": tbm,
                                    "FThreadblockShapeN": tbn,
                                    "FThreadblockShapeK": tbk,
                                    "FWarpShapeM": wsm,
                                    "FWarpShapeN": wsn,
                                    "FWarpShapeK": wsk,
                                    "Fused": True
                                }
                                datas.append(data)
    return datas


if __name__ == "__main__":
    datas_fused = generate_json(fused=True)
    json.dump(datas_fused, open("fused.json", "w"))
    datas_non_fused = generate_json(fused=False)
    json.dump(datas_non_fused, open("non_fused.json", "w"))
    datas_all = datas_fused + datas_non_fused
    json.dump(datas_all, open("all.json", "w"))

    datas_sample = np.random.choice(datas_all, 100, replace=False)
    json.dump(datas_sample.tolist(), open("sample.json", "w"))