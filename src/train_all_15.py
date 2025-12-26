import wandb

from train import get_default_config, train_mnk

if __name__ == "__main__":
    architectures = [
        "transformer_b_l",
        "transformer_b_s",
        "resnet_b_l",
        "resnet_b_s",
        "cnn_b_l",
        "cnn_b_s",
    ]

    for arch in architectures:
        config = get_default_config()
        config["architecture_name"] = arch
        config["mnk"] = (15, 15, 5)
        config["total_environment_steps"] = 600_000_000
        config["entropy_coef_schedule"]["params"]["total_steps"] = 300_000_000
        config["n_steps"] = 512
        config["batch_size"] = 4096

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
            group="main_run_15x15_board",
            name=f"run_{arch}_15x15",
            tags=[arch, "main_experiment", "15x15"],
        ) as run:
            train_mnk(run)
