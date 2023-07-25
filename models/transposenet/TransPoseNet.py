from models.model_base import BasePoseLightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from vit_pytorch import SimpleViT


class TransPoseNet(BasePoseLightningModule):
    #TODO: Add intro and referencce
    """
    TBD
    """
    def __init__(self, config):
        super(TransPoseNet, self).__init__(config)

        # Efficient net
        self.backbone_dim = 1280
        self.latent_dim = 1024

        # Position (t) and orientation (rot) encoders
        self.decoder_dim = config.get('decoder_dim')
        self.trans_vit = SimpleViT(image_size=14,
                                   patch_size=2,
                                   num_classes=1024,
                                   dim=256,
                                   depth=6,
                                   heads=4,
                                   channels=112,
                                   mlp_dim=1024
                                   )
        self.orient_vit = SimpleViT(image_size=28,
                                    patch_size=2,
                                    num_classes=1024,
                                    dim=256,
                                    depth=6,
                                    heads=4,
                                    channels=40,
                                    mlp_dim=1024
                                    )

        # Regressor layers
        # self.trans_fc_in = nn.Linear(256, self.latent_dim)
        self.trans_fc_out = nn.Linear(self.latent_dim, 3)
        # self.orient_fc_in = nn.Linear(256, self.latent_dim)
        self.orient_fc_out = nn.Linear(self.latent_dim, 4)

        self.trans_dropout = nn.Dropout(p=0.1)
        self.orient_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        features = self._backbone.extract_endpoints(x)
        src_t = features['reduction_4']
        src_rot = features['reduction_3']

        x_trans = self.trans_dropout(F.relu(self.trans_vit(src_t)))
        x_orient = self.orient_dropout(F.relu(self.orient_vit(src_rot)))

        p_x = self.trans_fc_out(x_trans)
        p_q = self.orient_fc_out(x_orient)

        return torch.cat((p_x, p_q), dim=1)

