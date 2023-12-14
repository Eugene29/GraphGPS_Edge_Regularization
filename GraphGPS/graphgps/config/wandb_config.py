from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_wandb')
def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration.
    """

    # WandB group
    cfg.wandb = CN()

    # Use wandb or not
    cfg.wandb.use = False
    
    cfg.reg = 0.0
    cfg.loss = ""
    # Wandb entity name, should exist beforehand
#     cfg.wandb.entity = "gtransformer" 
    cfg.wandb.entity = "kuetal"
#     cfg.wandb.entity = "euku"

    # Wandb project name, will be created in your team if doesn't exist already
    cfg.wandb.project = "gtblueprint"

    # Optional run name
    cfg.wandb.name = ""