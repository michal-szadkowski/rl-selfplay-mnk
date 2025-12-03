import wandb

from train import get_default_config, train_mnk

if __name__ == "__main__":
    architectures = ["transformer_s", "resnet_s", "cnn_s"]

    for arch in architectures:
        config = get_default_config()
        config["architecture_name"] = arch

        with wandb.init(
            config=config,
            project="mnk",
            group="main_run_small_board",
            name=f"run_{arch}",
            tags=[arch, "main_experiment"],
        ) as run:
            train_mnk(run)
