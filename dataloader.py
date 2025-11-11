import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import random_split,DataLoader, ConcatDataset
from PIL import Image
from skimage import io

class PMnet_usc(Dataset):
    def __init__(self,
                 dir_dataset="USC/",               
                 transform= transforms.ToTensor()):
        
        self.dir_dataset = dir_dataset
        self.transform = transform
        self.map_files = [f for f in os.listdir(self.dir_dataset+"map/") if f.endswith('.png')]
        self.tx_files = [f for f in os.listdir(self.dir_dataset+"Tx/") if f.endswith('.png')]
        # self.rx_files = [f for f in os.listdir(self.dir_dataset+"Rx/") if f.endswith('.png')]
        self.power_files = [f for f in os.listdir(self.dir_dataset+"pmap/") if f.endswith('.png')]

        self.map_paths = [os.path.join(self.dir_dataset+"map/", f) for f in self.map_files]
        self.tx_paths = [os.path.join(self.dir_dataset+"Tx/", f) for f in self.tx_files]
        # self.rx_paths = [os.path.join(self.dir_dataset+"Rx/", f) for f in self.rx_files]
        self.power_paths = [os.path.join(self.dir_dataset+"pmap/", f) for f in self.power_files]

    def __len__(self):
        return len(self.map_paths)
    
    def __getitem__(self, idx):
        target_size = (256,256)

        # Load city map
        map_path = self.map_paths[idx]
        
        image_map = np.array(Image.open(map_path).resize(target_size,Image.LANCZOS)) 
        # image_map = np.asarray(io.imread(map_path))

        # Load Tx
        tx_path = self.tx_paths[idx]
        image_tx = np.array(Image.open(tx_path).resize(target_size,Image.LANCZOS))
        # image_tx = np.asarray(io.imread(tx_path))

        # Load Rx 未被使用
        # rx_path = self.rx_paths[idx]
        # image_rx = np.asarray(io.imread(rx_path))

        # Load Power
        power_path = self.power_paths[idx]
        image_power = np.array(Image.open(power_path).resize(target_size,Image.LANCZOS))
        # image_power = np.asarray(io.imread(power_path))
      

        inputs=np.stack([image_map, image_tx], axis=2)

        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            power = self.transform(image_power).type(torch.float32)

        return [inputs , power]

# 获取数据加载器 数据集地址 训练集比例 批量大小 工作线程数
def get_loader(dir_dataset,train_ratio,test_ratio,batch_size,num_workers):
    
    data = PMnet_usc(dir_dataset)
    dataset_size = len(data)
    train_size = int(dataset_size * train_ratio)
    test_size = int(dataset_size * test_ratio)
    val_size = dataset_size - train_size - test_size    
    # 设置固定的随机种子
    generator = torch.Generator()
    generator.manual_seed(42)  # 使用固定的种子值42
    
    train_dataset, test_dataset, val_dataset = random_split(data, [train_size,test_size,val_size], generator=generator)
    
    # 修改DataLoader配置
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True
    )
    
    return train_loader, test_loader, val_loader

def get_loader_from_multiple_datasets(list_dir_datasets, train_ratio, test_ratio, batch_size, num_workers):
    # 用于确保拆分可复现的固定随机种子生成器
    generator = torch.Generator()
    generator.manual_seed(42)  # 使用固定的种子值42

    all_train_datasets = []
    all_test_datasets = []
    all_val_datasets = []

    # 1. 遍历每个数据集路径
    for dir_dataset in list_dir_datasets:
        # 获取单个数据集实例
        data = PMnet_usc(dir_dataset)
        dataset_size = len(data)
        
        # 计算拆分大小
        train_size = int(dataset_size * train_ratio)
        test_size = int(dataset_size * test_ratio)
        # 验证集大小确保总和精确等于原始数据集大小
        val_size = dataset_size - train_size - test_size
        
        # 2. 拆分单个数据集
        train_dataset, test_dataset, val_dataset = random_split(
            data, 
            [train_size, test_size, val_size], 
            generator=generator
        )
        
        # 3. 收集拆分后的数据集
        all_train_datasets.append(train_dataset)
        all_test_datasets.append(test_dataset)
        all_val_datasets.append(val_dataset)

    # 4. 合并所有相同类型的拆分数据集
    # 使用 ConcatDataset 将所有数据集的训练集、测试集、验证集合并
    if not all_train_datasets:
         # 如果列表为空，则返回空加载器或抛出错误
        raise RuntimeError("没有找到任何数据集来创建加载器。")

    merged_train_dataset = ConcatDataset(all_train_datasets)
    merged_test_dataset = ConcatDataset(all_test_datasets)
    merged_val_dataset = ConcatDataset(all_val_datasets)

    # 5. 创建最终的 DataLoader

    # 辅助配置，减少重复代码
    loader_kwargs = dict(
        batch_size=batch_size, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True
    )

    # 训练集通常需要打乱 (shuffle=True)
    train_loader = DataLoader(
        merged_train_dataset, 
        shuffle=True, 
        **loader_kwargs
    )
    
    # 测试集和验证集通常不需要打乱 (shuffle=False)
    # 为了和您的原代码保持一致，这里保留 shuffle=True，但通常建议改为 False
    test_loader = DataLoader(
        merged_test_dataset, 
        shuffle=True, # 建议改为 False
        **loader_kwargs
    )

    val_loader = DataLoader(
        merged_val_dataset, 
        shuffle=True, # 建议改为 False
        **loader_kwargs
    )
    
    return train_loader, test_loader, val_loader

from PIL import Image

def show_tensor_image(tensor, index):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    img = tensor[index]
    print(img.shape)

    if img.shape[0] == 1:
        img = img.squeeze(0)

    print(img.shape)


    img = img.detach().numpy()

    img = ((img - img.min()) * (255 / (img.max() - img.min()))).astype(np.uint8)

    img = Image.fromarray(img)
    img.show()


if __name__ == "__main__":
    train_loader, test_loader = get_loader(dir_dataset="dataset/USC/",train_ratio=0.8,batch_size=16,num_workers=4)
    for i, data in enumerate(train_loader):
        
        # show_tensor_image(data[0],0)
        show_tensor_image(data[1],0)
        # print(data[1][0])
        break


