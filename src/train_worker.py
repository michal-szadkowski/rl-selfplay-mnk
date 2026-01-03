import wandb
import sys
import os
from train import get_default_config, train_mnk


def run_training(arch, board_size):
    config = get_default_config()
    config["architecture_name"] = arch

    if board_size == "13x13":
        config["mnk"] = (13, 13, 5)
        config["total_environment_steps"] = 600_000_000
        config["entropy_coef_schedule"]["params"]["total_steps"] = 300_000_000
        config["batch_size"] = 4096
        group_name = "final"
    else:
        group_name = "final"

    if "transformer" in arch:
        config["entropy_coef_schedule"]["params"]["final_coef"] = 0.01
        config["entropy_coef"] = 0.10
        config["learning_rate"] = 12e-4
    elif "resnet" in arch:
        config["entropy_coef_schedule"]["params"]["final_coef"] = 0.001
        config["entropy_coef"] = 0.05
        config["learning_rate"] = 8e-4
    elif "cnn" in arch:
        config["entropy_coef_schedule"]["params"]["final_coef"] = 0.001
        config["entropy_coef"] = 0.04
        config["learning_rate"] = 6e-4

    with wandb.init(
        config=config,
        project="mnk_b",
        group=group_name,
        name=f"run_{arch}_{board_size}",
        tags=[arch, board_size, "final_final"],
    ) as run:
        train_mnk(run)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        run_training(sys.argv[1], sys.argv[2])
