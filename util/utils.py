import logging
import logging.config
import PIL
import json
from os.path import join, exists, split, realpath
import time
from os import makedirs, getcwd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from omegaconf import OmegaConf
import os
from fnmatch import fnmatch
from typing import List
import re
import torch.nn as nn
from collections import OrderedDict


##########################
# Logging and output utils
##########################
def get_stamp_from_log():
    """
    Get the time stamp from the log file
    :return:
    """
    return split(logging.getLogger().handlers[0].baseFilename)[-1].replace(".log", "")


def create_output_dir(out_dir):
    """
    Create a new directory for outputs, if it does not already exist
    :param out_dir: (str) the name of the directory
    :return: the path to the outpur directory
    """
    if not exists(out_dir):
        makedirs(out_dir, exist_ok=True)
    return out_dir


def save_config_to_output_dir(path, config) -> None:
    with open(join(path, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)


def init_logger(outpath: str = None, suffix: str = None) -> str:
    """
    Initialize the logger and create a time stamp for the file
    """
    path = split(realpath(__file__))[0]

    with open(join(path, 'log_config.json')) as json_file:
        log_config_dict = json.load(json_file)
        filename = log_config_dict.get('handlers').get('file_handler').get('filename')
        filename = ''.join([filename, "_", time.strftime("%d_%m_%y_%H_%M", time.localtime())])

        # Creating logs' folder is needed
        if outpath is not None:
            log_path = create_output_dir(join(outpath, filename))
        else:
            log_path = create_output_dir(join(getcwd(), 'out', filename))

        if suffix is not None:
            filename += suffix
        log_config_dict.get('handlers').get('file_handler')['filename'] = join(log_path, f'{filename}.log')
        logging.config.dictConfig(log_config_dict)

        # disable external modules' loggers (level warning and below)
        logging.getLogger(PIL.__name__).setLevel(logging.WARNING)
        return log_path


def get_checkpoint_list(path: str) -> List[str]:
    ckpts_list = {'ckpts': {}}
    for path, subdirs, files in os.walk(path):
        for name in sorted(files):
            if fnmatch(name, "*.ckpt"):
                m = re.match(".+_checkpoint-epoch_(\d+).ckpt", name)
                if m is not None:
                    ckpts_list['ckpts'][int(m.group(1))] = join(path, name)
                elif 'last' in name:
                    ckpts_list['last'] = join(path, name)

    return ckpts_list


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda'), dtypes=None):
    # Reference: https://github.com/sksq96/pytorch-summary
    if dtypes is None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = []

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str.append("----------------------------------------------------------------")
    line_new = "{:<20}  {:<25} {:<15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str.append(line_new)
    summary_str.append("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:<20}  {:<25} {:<15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str.append(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str.append("================================================================")
    summary_str.append("Total params: {0:,}".format(total_params))
    summary_str.append("Trainable params: {0:,}".format(trainable_params))
    summary_str.append("Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params))
    summary_str.append("----------------------------------------------------------------")
    summary_str.append("Input size (MB): %0.2f" % total_input_size)
    summary_str.append("Forward/backward pass size (MB): %0.2f" % total_output_size)
    summary_str.append("Params size (MB): %0.2f" % total_params_size)
    summary_str.append("Estimated Total Size (MB): %0.2f" % total_size)
    summary_str.append("----------------------------------------------------------------")
    # return summary
    return summary_str, (total_params, trainable_params)


##########################
# Evaluation utils
##########################
def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx7, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    posit_err = torch.norm(est_pose[:, 0:3] - gt_pose[:, 0:3], dim=1)
    est_pose_q = F.normalize(est_pose[:, 3:], p=2, dim=1)
    gt_pose_q = F.normalize(gt_pose[:, 3:], p=2, dim=1)
    inner_prod = torch.bmm(est_pose_q.view(est_pose_q.shape[0], 1, est_pose_q.shape[1]),
                           gt_pose_q.view(gt_pose_q.shape[0], gt_pose_q.shape[1], 1))
    orient_err = 2 * torch.acos(torch.abs(inner_prod)) * 180 / np.pi
    return posit_err, orient_err


##########################
# Plotting utils
##########################
def plot_loss_func(sample_count, loss_vals, loss_fig_path):
    plt.figure()
    plt.plot(sample_count, loss_vals)
    plt.grid()
    plt.title('Camera Pose Loss')
    plt.xlabel('Number of samples')
    plt.ylabel('Loss')
    plt.savefig(loss_fig_path)


##########################
# Augmentations
##########################
train_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

}
test_transforms = {
    'baseline': transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
                                    ])
}


##########################
# Evaluation utils
##########################
def freeze_model_components(model: torch.nn.Module, freeze_exclude: List[str]):
    for name, parameter in model.named_parameters():
        # Setting whether to freeze the parameter or not
        freeze_param = True
        for phrase in freeze_exclude:
            # Excluding parameter if specified
            if phrase in name:
                freeze_param = False
                break

        # Freezing parameter
        if freeze_param:
            parameter.requires_grad_(False)
