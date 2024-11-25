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
    m_search_space = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 3072]
    n_search_space = [64, 96, 128, 192, 384]
    
    INTERLEAVE = 64
    N = 384
    datas = []
    if not fused:
        for tbm in m_search_space:
            for tbn in n_search_space:
                for tbk in [128]:
                    for wsm in m_search_space:
                        if tbm % wsm != 0:
                            continue
                        for wsn in n_search_space:
                            if tbn % wsn != 0:
                                continue
                            for wsk in [128]:
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
        for tbm in m_search_space:
            for tbn in [N]:
                for tbk in [128]:
                    for wsm in m_search_space:
                        if tbm % wsm != 0:
                            continue
                        for wsn in n_search_space:
                            if tbn % wsn != 0:
                                continue
                            for wsk in [128]:
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


def generate():
    datas_fused = generate_json(fused=True)
    json.dump(datas_fused, open("fused.json", "w"))
    datas_non_fused = generate_json(fused=False)
    json.dump(datas_non_fused, open("non_fused.json", "w"))
    datas_all = datas_fused + datas_non_fused
    json.dump(datas_all, open("all.json", "w"))

    datas_sample = np.random.choice(datas_all, 100, replace=False)
    json.dump(datas_sample.tolist(), open("sample.json", "w"))
    
    print(f"Generated {len(datas_fused)} fused configs")
    print(f"Generated {len(datas_non_fused)} non-fused configs")
    print(f"Generated {len(datas_all)} all configs")

def get_factors(n):
    factors = []
    for i in range(1, n+1):
        if n % i == 0:
            factors.append(i)
    return factors

if __name__ == "__main__":
    # print(get_factors(384))
    # print(get_factors(12*256))
    generate()