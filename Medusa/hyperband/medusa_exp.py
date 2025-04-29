import numpy as np
from medusa.train.train_legacy import training_run

def get_params():
    medusa_num_heads = [2, 3, 4, 5]
    medusa_num_layers = [1, 2]
    return

def try_params(n_epochs, model_args):
    inference_time = training_run(
        n_epochs=n_epochs,
        model_args=model_args)
    return {
        "loss": inference_time,
        "params": model_args,
    }




if __name__ == "__main__":
    from hyperband.hyperband import Hyperband
    hb = Hyperband(get_params, try_params)
    results = hb.run()
    best = sorted(results, key=lambda x: x["loss"])[0]
    print("Best config:", best)