import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from config import config
from box import Box
import torch.nn as nn
from model import Model
import pandas as pd
from dataset import FishDataModule

import argparse

parser = argparse.ArgumentParser(description='Model path')
parser.add_argument('path', type=str,
                    help='Model path for evaluating')
args = parser.parse_args()
def criterion(outputs, targets):
    return torch.sqrt(nn.MSELoss()(outputs.view(-1), targets.view(-1)))
@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    TARGETS = []
    PREDS = []
    paths=[]
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (images, targets,im_path) in bar:   
        # print(im_path)     
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        PREDS.append(outputs.view(-1).cpu().detach().numpy())
        TARGETS.append(targets.view(-1).cpu().detach().numpy())
        paths.append(im_path)
        
        # bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
        #                 LR=optimizer.param_groups[0]['lr'])   
    
    TARGETS = np.concatenate(TARGETS)
    PREDS = np.concatenate(PREDS)
    paths=np.concatenate(paths)
    df=pd.DataFrame({'target':TARGETS,"predicted":PREDS,"image_path":paths})
    df.to_csv("test_predictions.csv")
    val_rmse = mean_squared_error(TARGETS, PREDS, squared=False)
    
    return epoch_loss, val_rmse
config=Box(config)
model = Model(config)
model.load_from_checkpoint(args.path)
model.eval()
device=model.device
test_df=pd.read_csv("test.csv")
test_dataloader = FishDataModule(test_df, test_df, config).val_dataloader()
valid_one_epoch(model,test_dataloader,device,1)