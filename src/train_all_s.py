import wandb

from train import get_default_config, train_mnk

if __name__ == "__main__":
    architectures = ["transformer_b_s", "resnet_b_s", "cnn_b_s"]

    for arch in architectures:
        config = get_default_config()
        config["architecture_name"] = arch
        config["learning_rate"] = 5e-4

        if "transformer" in arch:
            config["entropy_coef"] = 0.1
        else:
            config["entropy_coef"] = 0.04

        with wandb.init(
            config=config,
            project="mnk_b",
            group="main_run_small_board_s",
            name=f"run_{arch}",
            tags=[arch, "main_experiment"],
        ) as run:
            train_mnk(run)
