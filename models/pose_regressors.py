from .model_base import BasePoseLightningModule
from models.posenet.PoseNet import PoseNet
from models.transposenet.TransPoseNet import TransPoseNet


def get_model(model_name, config) -> BasePoseLightningModule:
    if model_name == 'posenet':
        return PoseNet(config)
    if model_name == 'transposenet':
        return TransPoseNet(config)
    else:
        raise "{} not supported".format(model_name)


def load_model_from_checkpoint(model_name, ckpt_path) -> BasePoseLightningModule:
    if model_name == 'posenet':
        return PoseNet.load_from_checkpoint(ckpt_path)
    if model_name == 'transposenet':
        return TransPoseNet.load_from_checkpoint(ckpt_path)
    else:
        raise "{} not supported".format(model_name)