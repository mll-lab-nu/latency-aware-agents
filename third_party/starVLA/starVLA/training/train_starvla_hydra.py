import hydra
from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.train_starvla import main


@hydra.main(version_base="1.1", config_path="../../examples/rl_games/config", config_name="train")
def hydra_main(cfg):
    cfg = apply_config_compat(cfg)
    main(cfg)


if __name__ == "__main__":
    hydra_main()
