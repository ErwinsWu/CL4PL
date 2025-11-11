import torch
import torch.nn as nn
import time
import numpy as np
from utils import *

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

model_name = 'LRRA'
tasks = ['USC','Boston','UCLA']
task = 'UCLA'
dataset_name = 'RadioMap'

# 实例化模型并转到GPU（如果有）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(model_name,tasks,dataset_name)
model.to(device)
model.eval()

# 计算模型大小
def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

print(f"Model size: {get_model_size(model):.2f} MB")

# 测试推理时间
def measure_inference_time(model, input_size=(1,2,256,256), repeat=100):
    x = torch.randn(*input_size).to(device)
    
    # 先预热几次
    for _ in range(10):
        _ = model_out(x,model_name,model,task)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(repeat):
        _ = model_out(x,model_name,model,task)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / repeat
    return avg_time

avg_infer_time = measure_inference_time(model)
print(f"Average inference time per batch: {avg_infer_time*1000:.3f} ms")