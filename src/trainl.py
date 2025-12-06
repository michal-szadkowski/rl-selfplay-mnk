import wandb

from train import get_default_config, train_mnk

if __name__ == "__main__":
    architectures = ["transformer_l", "resnet_l", "cnn_l"]

    for arch in architectures:
        config = get_default_config()
        config["architecture_name"] = arch
        config["learning_rate"] = 7e-4

        if arch == "transformer_l":
            config["entropy_coef"] = 0.1
        else:
            config["entropy_coef"] = 0.04

        with wandb.init(
            config=config,
            project="mnk",
            group="main_run_small_board_l",
            name=f"run_{arch}",
            tags=[arch, "main_experiment"],
        ) as run:
            train_mnk(run)
