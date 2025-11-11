import argparse
import argparse
from utils import get_config, setup_logging, create_model
from dataloader import get_loader
import os
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

import torch
import torch.nn.functional as F

def ssim_pytorch(img1, img2, window_size=7, data_range=1.0):
    """PyTorch版本的SSIM，支持批量GPU计算"""
    # img1, img2: [B, C, H, W]
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # 创建高斯窗口
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                          for x in range(window_size)])
    window = gauss / gauss.sum()
    window = window.unsqueeze(0).unsqueeze(0)
    window = window.to(img1.device)
    
    # 计算均值
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2) - mu1_mu2
    
    # SSIM计算
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def calculate_metrics_gpu(y_true, y_pred, stats=None):
    """GPU加速版本"""
    if stats is None:
        stats = {
            'n_samples': 0,
            'sum_se': 0.0,
            'sum_ae': 0.0,
            'sum_y': 0.0,
            'sum_y2': 0.0,
            'ssim_sum': 0.0,
            'ssim_count': 0
        }
    
    # 保持在GPU上计算
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    diff = y_true_flat - y_pred_flat
    stats['sum_se'] += torch.sum(diff ** 2).item()
    stats['sum_ae'] += torch.sum(torch.abs(diff)).item()
    stats['sum_y'] += torch.sum(y_true_flat).item()
    stats['sum_y2'] += torch.sum(y_true_flat ** 2).item()
    stats['n_samples'] += y_true_flat.numel()
    
    # GPU批量SSIM（假设shape: [B, 1, H, W]）
    if len(y_true.shape) == 4:
        ssim_val = ssim_pytorch(y_true, y_pred)
        stats['ssim_sum'] += ssim_val.item()
        stats['ssim_count'] += 1
    
    return stats

def calculate_metrics_incremental(y_true, y_pred, stats=None):
    """增量更新统计量，避免存储所有数据"""
    
    # 转换为numpy
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # 初始化统计字典
    if stats is None:
        stats = {
            'n_samples': 0,
            'sum_se': 0.0,      # sum of squared errors
            'sum_ae': 0.0,      # sum of absolute errors
            'sum_y': 0.0,       # sum of true values
            'sum_y2': 0.0,      # sum of squared true values
            'sum_yy_pred': 0.0, # sum of (y_true * y_pred)
            'sum_pred2': 0.0,   # sum of squared predictions
            'ssim_sum': 0.0,
            'ssim_count': 0
        }
    
    # 展平计算
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # 更新统计量（向量化操作）
    diff = y_true_flat - y_pred_flat
    stats['sum_se'] += np.sum(diff ** 2)
    stats['sum_ae'] += np.sum(np.abs(diff))
    stats['sum_y'] += np.sum(y_true_flat)
    stats['sum_y2'] += np.sum(y_true_flat ** 2)
    stats['sum_yy_pred'] += np.sum(y_true_flat * y_pred_flat)
    stats['sum_pred2'] += np.sum(y_pred_flat ** 2)
    stats['n_samples'] += len(y_true_flat)
    
    # 批量计算SSIM（如果是图像数据）
    batch_size = y_true.shape[0]
    for i in range(batch_size):
        true_img = np.squeeze(y_true[i])
        pred_img = np.squeeze(y_pred[i])
        
        try:
            ssim_val = ssim(true_img, pred_img, data_range=1.0, win_size=7)
            if not np.isnan(ssim_val):
                stats['ssim_sum'] += ssim_val
                stats['ssim_count'] += 1
        except (ValueError, RuntimeWarning):
            continue
    
    return stats


def finalize_metrics(stats):
    """根据累积统计量计算最终指标"""
    n = stats['n_samples']
    
    # MSE & RMSE
    mse = stats['sum_se'] / n
    rmse = np.sqrt(mse)
    
    # MAE
    mae = stats['sum_ae'] / n
    
    # R² score
    mean_y = stats['sum_y'] / n
    ss_tot = stats['sum_y2'] - n * mean_y ** 2
    ss_res = stats['sum_se']
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # SSIM
    mean_ssim = stats['ssim_sum'] / stats['ssim_count'] if stats['ssim_count'] > 0 else 0.0
    
    # PSNR
    psnr = 20 * np.log10(1.0) - 10 * np.log10(mse) if mse > 0 else float('inf')
    
    return {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'SSIM': float(mean_ssim),
        'PSNR': float(psnr)
    }


def evaluate_model(model, dataset, dataset_name, eval_config, logger, model_name, task=None):
    checkpoint_path = eval_config['model_path']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    test_loader = dataset['test']
    model_path = os.path.join(checkpoint_path, f"UCLA_Boston_best_model.pth")
    
    # model.load_state_dict(torch.load(model_path, weights_only=False), strict=False)
    model.load_state_dict(torch.load(model_path,map_location=device), strict=False)
    logger.info(f"model_path: {model_path}")
    model.eval()

    # 增量统计
    stats = None
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model_out(inputs, model_name, model, task)
            
            # 增量更新（每个batch处理后立即释放）
            stats = calculate_metrics_incremental(targets, outputs, stats)
    
    # 计算最终指标
    metrics = finalize_metrics(stats)
    logger.info(f"Test Metrics: {metrics}")
    print(f"Test Metrics: {metrics}")
    
    return metrics


def model_out(inputs,model_name,model,task=None):
    if model_name == 'DCLSE':
        inputs_dic = {
            'tensor':inputs,
            'task':task
        }
        outputs = model(inputs_dic)
    elif model_name == 'LRRA':
        inputs_dic = (inputs,task)
        outputs = model(inputs_dic)
    elif model_name == 'FE':
        # inputs_dic = (inputs,task)
        outputs = model(inputs,task)
    elif model_name == 'DISTILL':
        outputs = model(inputs)
    elif model_name == 'PARALLEL' or model_name == 'SERIES' or model_name == 'RCM':
        outputs = model((inputs,task))
    else:
        outputs = model(inputs)

    return outputs

def main():
    # load config
    parser = argparse.ArgumentParser()
    # vanilla.yaml
    parser.add_argument('--config', type=str, default='configs/ewc.yaml',help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    model_config = config['model']
    dataset_config = config['dataset']
    eval_config = config['evaluation']

    # set logger
    logger = setup_logging(os.path.join(config['logs']['test_logger_path'],f'{dataset_config["dataset_name"]}_test.log'))

    # create model
    logger.info(f"Creating model: {model_config['model_name']}")
    model = create_model(model_config['model_name'], model_config['tasks'],dataset_config['dataset_name'])

    # load dataset 
    logger.info(f"Loading dataset from: {dataset_config['dataset']}")

    train_loader, test_loader, val_loader = get_loader(
            dir_dataset=dataset_config['dataset'],
            batch_size=dataset_config['batch_size'],
            train_ratio=dataset_config['train_ratio'],
            test_ratio=dataset_config['test_ratio'],
            num_workers=dataset_config['num_workers'])

    
    dataset = {
            'train': train_loader,
            'test': test_loader,
            'val': val_loader
        }

    dataset_name = dataset_config['dataset'].split('/')[-2]
    task = None
    if 'task' in model_config:
        task = model_config['task']

    logger.info("Starting evaluation...")
    evaluate_model(model, dataset, dataset_name, eval_config, logger,model_config['model_name'],task=task)

if __name__ == "__main__":
    main()