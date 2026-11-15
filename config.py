"""Configuration transcribed from the author's V1 run settings."""
from argparse import Namespace


def model_args():
    return Namespace(
        input_dim=768, emb_dim=384, n_classes=2, n_layers=2,
        cluster_number=5, alpha=0.1, drop_out=False, merge_token=True,
        pmt_clu=True, moving_pmt=True, momentum=0.01,
        multiscale=True, multiscale_norm=False, spatial_mode="both",
        pos_enc=False, feature_norm=False,
        feature_suffix="_high_feature_ctrans", feature_format="pkl",
        cluster_mode="original", cluster_temp=0.1, occ_rho=0.02, occ_weight=0,
    )


MAX_EPOCHS = 300
MIN_EPOCHS = 20
PATIENCE = 20
FUSION_LR = 2e-4
BACKBONE_LR = 2e-5
WEIGHT_DECAY = 1e-5
