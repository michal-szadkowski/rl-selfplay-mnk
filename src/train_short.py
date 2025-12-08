import wandb

from train import get_default_config, train_mnk

if __name__ == "__main__":
    config = get_default_config()

    config["total_environment_steps"] = 80_000_000
    config["entropy_coef_schedule"] = {
        "type": "linear",
        "params": {"final_coef": 0.001, "total_steps": 50_000_000},
    }
    config["lr_decay"] = False

    with wandb.init(config=config, project="mnk_b_sweeps") as run:
        train_mnk(run)
