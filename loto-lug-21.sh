#!/bin/bash
source .venv/bin/activate
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=7 --tube=0 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=7 --tube=1 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=7 --tube=2 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=7 --tube=3 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=7 --tube=4 --arch=2d

python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=14 --tube=0 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=14 --tube=1 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=14 --tube=2 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=14 --tube=3 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=14 --tube=4 --arch=2d

python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=21 --tube=0 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=21 --tube=1 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=21 --tube=2 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=21 --tube=3 --arch=2d
python crossval_perexp.py --config_path=configs/baseline_jul21.yaml --tag=lotocv --discard_results=true --seed=21 --tube=4 --arch=2d
