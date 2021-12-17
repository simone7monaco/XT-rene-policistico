from crossval_perexp import main as launch_cv
from pathlib import Path
from easydict import EasyDict as ed
import sys

start = int(sys.argv[1])
stop = int(sys.argv[2])

scripts = []
with open('loeo.txt', 'r') as f:
    for i, line in enumerate(f):
        if i < stop and i >= start:
            scripts.append(line[:-1])
            
# crossval_perexp.py --alternative_model=pspnet --config_path=configs/baseline.yaml --discard_results=true --eval_network=true --seed=7 --test_tube=5

for s in scripts:
    s = s[18:].split(" ")
    model = [t.split("=")[-1] for t in s if "model" in t][0]
    seed = [t.split("=")[-1] for t in s if "seed" in t][0]
    try:
        tube = [t.split("=")[-1] for t in s if "tube" in t][0]
    except:
        exp = [t.split("=")[-1] for t in s if "exp" in t][0]
    tag = [t.split("=")[-1] for t in s if "tag" in t][0]

    launch_cv(
        ed({"alternative_model": model,
        "config_path": Path('configs/baseline.yaml'), 
        "dataset": 'latest',
        "tag": tag,
        "discard_results": True,
        "exp": int(exp) if 'exp' in locals() else None,
        "focus_size": None,
        "k": 0,
        "seed": int(seed),
        "stratify_fold": False,
        "tube": int(tube) if 'tube' in locals() else None,
        "tiling": False,
        "single_exp": None
       })
    )
    print("\n\n\n\n\n")
