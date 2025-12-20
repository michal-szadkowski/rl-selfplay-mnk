import wandb

from train import get_default_config, train_mnk

if __name__ == "__main__":
    architectures = [
        "transformer_b_s",
        "resnset_b_s",
        "cnn_b_s",
    ]

    for arch in architectures:
        config = get_default_config()
        config["architecture_name"] = arch

        config["entropy_coef_schedule"]["params"]["final_coef"] = 0.01

        if "transformer" in arch:
            config["entropy_coef"] = 0.10
            config["learning_rate"] = 12e-4
        elif "resnet" in arch:
            config["entropy_coef"] = 0.05
            config["learning_rate"] = 8e-4
        elif "cnn" in arch:
            config["entropy_coef"] = 0.04
            config["learning_rate"] = 6e-4

        with wandb.init(
            config=config,
            project="mnk_b",
            group="main_run2_small_board",
            name=f"run4_{arch}",
            tags=[arch, "main_experiment"],
            notes="Zwiększony parametr entropy final_coef",
        ) as run:
            train_mnk(run)
