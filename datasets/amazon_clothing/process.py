import os
import json

inp_path = "raw/Clothing_Shoes_and_Jewelry_5.json"
out_path = "processed/train.txt"

with open(inp_path, "r") as f:
    lines = f.readlines()

f_out = open(out_path, "w")
for line in lines:
    d = json.loads(line)
    f_out.write(d["reviewText"].replace("\n", "").strip() + "\n")

f_out.close()


