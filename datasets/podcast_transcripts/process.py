import os
import json
from tqdm import tqdm

inp_paths = [os.path.join("raw", file) for file in os.listdir("raw") if "json" in file]
out_paths = [
    os.path.join("processed", part + ".txt") for part in [
        sub[:sub.find("-")] for sub in os.listdir("raw") if "json" in sub]
]

for inp_path, out_path in zip(inp_paths, out_paths):

    with open(inp_path, "r") as f:
        d = json.load(f)

    f_out = open(out_path, "w")

    for k in tqdm(d.keys()):
        for el in d[k]:
            txt = el["utterance"].replace("\n", "").strip()
            f_out.write(txt + "\n")

    f_out.close()


