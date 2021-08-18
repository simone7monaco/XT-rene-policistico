# import neptune.new as neptune

# import neptune
# import skopt
from pathlib import Path
import numpy as np
from train import train

import matplotlib.pyplot as plt
import skopt.plots
import neptunecontrib.monitoring.skopt as sk_utils

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler

# from wandb.sweeps.config import tune
# from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch
# from wandb.sweeps.config.hyperopt import hp


def single_train(config):
    try:
        result = train(config)
    except Exception as e:
        print(e)
        result = 0
    
    return result

parameters = {
        "config_path": Path.home() / "my_rene-policistico/configs/baseline.yaml",
        "alternative_model": "unetpp",
        "lr": tune.uniform(1e-6, 1e-2),
        "batch_size": tune.choice([4, 8, 16, 32]),
        "acc_grad": tune.choice([1, 2, 4, 8, 16]),
        "optimizer": None,
        "scheduler": None,
        "b_search": True,
        "dataset": "latest",
        "wandb": {
            "project": "comparison", 
            "entity": "rene-policistico",
        }
}

hyperopt_alg = HyperOptSearch(metric="is_score",mode="max")
hyperopt_alg = ConcurrencyLimiter(hyperopt_alg, max_concurrent=2)

sha_schedular = AsyncHyperBandScheduler(metric="is_score",
                            mode="max",max_t=300)

analysis = tune.run(
    train,
    search_alg = hyperopt_alg, # Specify the search algorithm
#     resources_per_trial={'gpu': 1},
    num_samples=15,
    config=parameters,
    verbose=2
    )