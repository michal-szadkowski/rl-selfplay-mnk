import wandb

from train import get_default_config, train_mnk

if __name__ == "__main__":
    config = get_default_config()
    config["total_environment_steps"] = 40_000_000
    config["entropy_coef_schedule"] = (
        {
            "type": "linear",
            "params": {"final_coef": 0.001, "total_steps": 20_000_000},
        },
    )
    with wandb.init(
        config=config,
        project="mnk-sweeps",
        tags=["sweep"],
    ) as run:
        train_mnk(run)
