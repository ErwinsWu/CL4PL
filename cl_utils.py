import torch
import torch.nn as nn
from tqdm import tqdm

class EWC:
    def __init__(self, model, dataloader, device='cpu', criterion=nn.MSELoss()):
        """
        EWC类初始化
        
        参数:
            model: 神经网络模型
            dataloader: 用于计算Fisher信息矩阵的数据加载器
            device: 计算设备 ('cpu' 或 'cuda')
            criterion: 损失函数，默认为MSE
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.criterion = criterion
        
        # 存储参数的重要性(Fisher信息矩阵)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.means = {}
        self.fisher = {}
        
        # 存储初始参数
        for n, p in self.params.items():
            self.means[n] = p.data.clone()
    
    def compute_fisher(self):
        """
        计算Fisher信息矩阵
        """
        self.model.eval()
        
        # 初始化Fisher信息矩阵
        for n, p in self.params.items():
            self.fisher[n] = torch.zeros_like(p.data)
        
        # 遍历数据计算梯度平方的期望
        # for inputs, targets in self.dataloader:
        for inputs, targets in tqdm(self.dataloader,desc="EWC compute fisher"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # 累加梯度平方
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data.pow(2) / len(self.dataloader)
    
    def penalty(self):
        """
        计算EWC惩罚项
        """
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.means:
                # 计算当前参数与旧参数的差异，乘以Fisher信息矩阵
                loss += (self.fisher[n] * (p - self.means[n]).pow(2)).sum()
        return loss


class MAS:
    def __init__(self, model, dataloader, device='cpu'):
        """
        MAS类初始化
        
        参数:
            model: 神经网络模型
            dataloader: 用于计算参数重要性的数据加载器
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.model = model
        self.dataloader = dataloader
        self.device = device
        
        # 存储参数的重要性(Omega)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.means = {}
        self.omega = {}
        
        # 存储初始参数
        for n, p in self.params.items():
            self.means[n] = p.data.clone()
            self.omega[n] = torch.zeros_like(p.data)
    
    def compute_omega(self):
        """
        计算参数重要性(Omega)
        """
        self.model.eval()
        
        # 遍历数据计算参数重要性
        # for inputs, _ in self.dataloader:  # MAS不需要标签
        for inputs, _ in tqdm(self.dataloader,desc="MAS compute omega"):
        
            inputs = inputs.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # MAS使用模型输出的L2范数作为重要性度量
            # 计算输出对参数的梯度
            output_norm = torch.norm(outputs, p=2, dim=1).mean()
            output_norm.backward()
            
            # 累加梯度绝对值
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    self.omega[n] += torch.abs(p.grad.data) / len(self.dataloader)
    
    def penalty(self):
        """
        计算MAS惩罚项
        """
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.means:
                # 计算当前参数与旧参数的差异，乘以重要性权重
                loss += (self.omega[n] * (p - self.means[n]).pow(2)).sum()
        return loss
    
