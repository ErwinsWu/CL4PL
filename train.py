import argparse
from utils import get_config, setup_logging, create_model,create_folder,load_pretrained_model,freeze_model
from dataloader import get_loader
from utils import kd_criterion,encoder_criterion
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time

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
    elif model_name == 'PARALLEL':
        outputs = model(inputs,task)
    else:
        outputs = model(inputs)

    return outputs

def train_model(model, dataset, dataset_name, training_config, logger,model_name,last_epoch=None,task=None,cl=None,lambda_cl=1000,tea_model=None,lambda_kd=0.5):
    num_epochs = training_config['num_epochs']
    checkpoint_path = training_config['checkpoint_path']
    # define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)


    train_loader = dataset['train']
    val_loader = dataset['val']

    # define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['lr'])
    if training_config['loss'] == 'MSE':
        ceriterion = torch.nn.MSELoss()
    
    # define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_loss = float('inf')

    # training loop
    if last_epoch is not None:
        first_epoch = last_epoch-1
        model_path = training_config['model_path']
        # model.load_state_dict(model_path)
        model = load_pretrained_model(model,model_path)
        best_val_loss = float(training_config['best_val_loss'])
    else:
        first_epoch = 0
    for epoch in range(first_epoch,num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            if cl is not None :
                loss += lambda_cl * cl.penalty()
            if tea_model is not None:
                tea_encoder_o,x1,x2,x3,x4,x5 = tea_model.encoder(inputs)
                tea_o = tea_model.decoder(tea_encoder_o,(inputs,x1,x2,x3,x4,x5))
                stu_encoder_o,x1,x2,x3,x4,x5 = model.encoder(inputs)
                stu_o = model.decoder(stu_encoder_o,(inputs,x1,x2,x3,x4,x5))
                loss = (1-lambda_kd) * ceriterion(stu_o,targets) + lambda_kd * encoder_criterion(stu_encoder_o,tea_encoder_o) + lambda_kd * kd_criterion(stu_o,tea_o)
                loss = loss.mean()
            else:
                outputs = model_out(inputs,model_name,model,task)
                # outputs = model(inputs) 
                loss = ceriterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # eval
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model_out(inputs,model_name,model,task)
                loss = ceriterion(outputs, targets)
                val_loss += loss.item()
            val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)
        # save model checkpoint
        if val_loss < best_val_loss:
            logger.info(f"Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f} -> {val_loss:.4f}")
            best_val_loss = val_loss
            create_folder(checkpoint_path)
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f"{dataset_name}_best_model.pth"))


def main():
    start_time = time.time()
    # load config
    parser = argparse.ArgumentParser()
    # vanilla.yaml
    parser.add_argument('--config', type=str, default='configs/parallel.yaml',help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    model_config = config['model']
    dataset_config = config['dataset']
    training_config = config['training']

    # set logger
    logger = setup_logging(os.path.join(config['logs']['train_logger_path'],f'{dataset_config["dataset_name"]}_train.log'))

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

    dataset_name = dataset_config['dataset'].split('/')[-2]  # Extract dataset name from path
    task = model_config['task']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cl = None
    # if model_config['model_name'] == 'EWC':
    if 'last_dataset' in dataset_config:
        train_loader, test_loader, val_loader = get_loader(
        dir_dataset=dataset_config['last_dataset'],
        batch_size=dataset_config['batch_size'],
        train_ratio=dataset_config['train_ratio'],
        test_ratio=dataset_config['test_ratio'],
        num_workers=dataset_config['num_workers'])

        last_dataset = {
                'train': train_loader,
                'test': test_loader,
                'val': val_loader
            }
        model.load_state_dict(torch.load(training_config['model_path']))
        model.to(device)
        if model_config['model_name'] == 'EWC':
            from cl_utils import EWC
            cl = EWC(model,last_dataset['train'],device,nn.MSELoss())
            cl.compute_fisher()
        elif model_config['model_name'] == 'MAS':
            from cl_utils import MAS
            cl = MAS(model,last_dataset['train',device,nn.MSELoss()])
            cl.compute_omega()


    if training_config['continue_train']:
        last_epoch = training_config['last_epoch']
        train_model(model, dataset, dataset_name, training_config, logger,model_config['model_name'], last_epoch=last_epoch,task=task)
        return # Exit after continuing training

    if model_config['model_name'] == 'LRRA' or model_config['model_name'] == 'FE' or model_config['model_name'] == 'PARALLEL':
        
        model_path = 'checkpoints/vanilla/USC_best_model.pth'
        # load the pretrained VANILLA model
        mapping = 'vanilla2new'
        model = load_pretrained_model(model,model_path,mapping=mapping)

        freeze_model(model,task='Boston')

    if model_config['model_name'] == 'DISTILL':
        last_model_path = model_config['last_model_path']
        mapping = 'distill'
        tea_model = create_model(model_config['model_name'])
        tea_model = load_pretrained_model(tea_model,last_model_path,mapping)
        logger.info("Start kd training...")
        train_model(model, dataset, dataset_name, training_config, logger,model_config['model_name'],task=task,cl=cl,stu_model=tea_model,lambda_kd=model_config['lambda_kd'])
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total training time: {elapsed_time/60:.2f} minutes")
        logger.info("Training completed.")
        return

    # train model
    logger.info("Starting training...")
    # task = 'USC'
    train_model(model, dataset, dataset_name, training_config, logger,model_config['model_name'],task=task,cl=cl)
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total training time: {elapsed_time/60:.2f} minutes")
    logger.info("Training completed.")

if __name__ == "__main__":
    main()