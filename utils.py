import yaml
import torch
import numpy as np
import os
import torch.distributed as dist
import logging

# 获取配置文件
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)
    
# 设置随机种子以确保可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def setup_logging(filename):

    log_dir = os.path.dirname(filename)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"创建日志目录: {log_dir}")

    logging.basicConfig(
        filename=filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger = logging.getLogger(__name__)

    return logger

# 创建文件路径
def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"创建目录: {folder_path}")
    
    return folder_path


def create_model(model_name,tasks=None,dataset_name=None):
    n_blocks = [3, 3, 27, 3]  
    atrous_rates = [6, 12, 18]  
    multi_grids = [1, 2, 4]  
    output_stride = 8 
    if model_name == 'VANILLA' or model_name == 'EWC' or model_name == 'MAS':
        from model.pmnet import PMNet
        model = PMNet(n_blocks, atrous_rates, multi_grids, output_stride)
    elif model_name == 'LRRA':
        from model.pmnet_lrra import PMNet
        model = PMNet(n_blocks, atrous_rates, multi_grids, output_stride)
    elif model_name == 'RCM':
        from model.pmnet_rcm import PMNet
        model = PMNet(n_blocks, atrous_rates, multi_grids, output_stride,tasks)
    elif model_name == 'PARALLEL':
        from model.pmnet_parallel import PMNet
        model = PMNet(n_blocks, atrous_rates, multi_grids, output_stride,tasks)
    elif model_name == 'SERIES':
        from model.pmnet_series import PMNet
        model = PMNet(n_blocks, atrous_rates, multi_grids, output_stride,tasks)
    elif model_name == "FE":
        from model.pmnet_fe import PMNet
        model = PMNet(n_blocks, atrous_rates, multi_grids, output_stride,tasks)
    elif model_name == 'DISTILL':
        from model.pmnet_distill import PMNet
        model = PMNet(n_blocks, atrous_rates, multi_grids, output_stride)
    else:
        raise ValueError(f"未知的模型名称: {model_name}")

    return model

import torch.nn.functional as F

def kd_criterion(student_outputs, teacher_outputs, reduction='batchmean'):
    """
    计算知识蒸馏损失。
    """
    """
    计算两个形状为 [batch_size, c, w, h] 的张量之间的KL散度
    p_logits: 目标分布的 logits (不需要经过 softmax)
    q_logits: 预测分布的 logits (不需要经过 softmax)
    """
    # 对 channel 维度进行 softmax 得到概率分布
    p = F.softmax(teacher_outputs, dim=1)
    
    log_q = F.log_softmax(student_outputs, dim=1)  # 直接计算 log softmax，数值稳定

    # 使用 KLDivLoss，注意它的输入是 log_Q，target 是 P
    # 所以形式是 D_KL(P || Q)
    kl = F.kl_div(log_q, p, reduction=reduction)

    return kl




def encoder_criterion(student_features, teacher_features, eps=1e-8):
    """
    计算编码器特征的损失。
    """
    stu = student_features.view(student_features.size(0), -1)
    tea = teacher_features.view(teacher_features.size(0), -1)

    stu_norm = F.normalize(stu, dim=1, eps=eps)
    tea_norm = F.normalize(tea, dim=1, eps=eps)

    sim = (stu_norm * tea_norm).sum(dim=1)  # [B]
    loss = 1 - sim  # [B]
    return loss

def load_pretrained_model(model, model_path,mapping=None):
    if mapping == 'vanilla2new':
        old_state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = model.state_dict()

        # 创建映射字典：旧参数名 -> 新参数名
        mapping_dict = {}
        
        # 1. 处理主干网络参数（没有适配器的部分）
        for key in old_state_dict:
            if key in new_state_dict:
                mapping_dict[key] = key
        
        # 2. 处理解码器参数（添加decoder.USC.前缀）
        for key in old_state_dict:
            if key.startswith('conv_up'):
                new_key = f'decoder.USC.{key}'
                if new_key in new_state_dict:
                    mapping_dict[key] = new_key
        
        # 3. 加载匹配的参数
        for old_key, new_key in mapping_dict.items():
            if old_state_dict[old_key].shape == new_state_dict[new_key].shape:
                new_state_dict[new_key] = old_state_dict[old_key]
            else:
                print(f"形状不匹配: {old_key} -> {new_key}")
        
        # 4. 加载新模型（适配器参数将保持随机初始化）
        model.load_state_dict(new_state_dict, strict=False)

    elif mapping == 'ft' or mapping == 'load':
        model.load_state_dict(torch.load(model_path))
    elif mapping == 'distill':
        old_state_dict = torch.load(model_path, map_location='cpu')
        new_state_dict = model.state_dict()

        # 创建映射字典：旧参数名 -> 新参数名
        mapping_dict = {}

        for key in old_state_dict:
            if key.startswith('conv_up'):
                new_key = f'decoder.{key}'
                if new_key in new_state_dict:
                    mapping_dict[key] = new_key
            if key.startswith('layer'):
                new_key = f'encoder.{key}'
                if new_key in new_state_dict:
                    mapping_dict[key] = new_key
            if key.startswith('aspp'):
                new_key = f'encoder.{key}'
                if new_key in new_state_dict:
                    mapping_dict[key] = new_key
            if key.startswith('fc1'):
                new_key = f'encoder.{key}'
                if new_key in new_state_dict:
                    mapping_dict[key] = new_key
            if key.startswith('reduce'):
                new_key = f'encoder.{key}'
                if new_key in new_state_dict:
                    mapping_dict[key] = new_key

        # 3. 加载匹配的参数
        for old_key, new_key in mapping_dict.items():
            if old_state_dict[old_key].shape == new_state_dict[new_key].shape:
                new_state_dict[new_key] = old_state_dict[old_key]
            else:
                print(f"形状不匹配: {old_key} -> {new_key}")
        
        # 4. 加载新模型（适配器参数将保持随机初始化）
        model.load_state_dict(new_state_dict, strict=False)
    
    return model

def freeze_model(model,task):
    for name, param in model.named_parameters():
        if task not in name :  # 只冻结不包含 'decoder.USC' 的参数
            param.requires_grad = False
            # print(f"冻结参数: {name}")
        else:
            # print(f"保留参数: {name}")
            pass