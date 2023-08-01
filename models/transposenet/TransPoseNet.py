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

        self.config = config

        # The projection of the activation map before going into the Transformer's encoder
        self.trans_in_encoder = nn.Conv2d(112, self.config.get('decoder_dim'), kernel_size=1)
        self.orient_encoder = nn.Conv2d(40, self.config.get('decoder_dim'), kernel_size=1)

        # Translation and orientation encoders
        self.decoder_dim = config.get('decoder_dim')
        self.trans_vit = SimpleViT(image_size=14,
                                   patch_size=1,
                                   num_classes=self.config.get('latent_dim'),
                                   dim=self.config.get('decoder_dim'),
                                   depth=self.config.get('depth'),
                                   heads=self.config.get('heads'),
                                   channels=self.config.get('decoder_dim'),
                                   mlp_dim=self.config.get('decoder_dim')
                                   )
        self.orient_vit = SimpleViT(image_size=28,
                                    patch_size=1,
                                    num_classes=self.config.get('latent_dim'),
                                    dim=self.config.get('decoder_dim'),
                                    depth=self.config.get('depth'),
                                    heads=self.config.get('heads'),
                                    channels=self.config.get('decoder_dim'),
                                    mlp_dim=self.config.get('decoder_dim')
                                    )

        # Regressor layers
        self.trans_fc = nn.Linear(self.config.get('latent_dim'), 3)
        self.orient_fc = nn.Linear(self.config.get('latent_dim'), 4)

    def forward(self, x):
        # Extracting feature maps from the Backbone's forward pass
        features = self._backbone.extract_endpoints(x)
        src_t = features['reduction_4']
        src_orient = features['reduction_3']

        # Translation branch forward pass
        trans_input = self.trans_in_encoder(src_t)
        x_trans = F.gelu(self.trans_vit(trans_input))
        p_x = self.trans_fc(x_trans)

        # Orientation branch forward pass
        orient_input = self.orient_encoder(src_orient)
        x_orient = F.gelu(self.orient_vit(orient_input))
        p_q = self.orient_fc(x_orient)

        return torch.cat((p_x, p_q), dim=1)

