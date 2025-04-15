# coding=utf-8
import argparse
import os
import sys
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from fno import FNO1d, FNO2d, FNO3d
from utils import *
from loss import darcy_loss,burger_1d_loss,ns_2d_loss

def get_model(pde_name):
    
    if pde_name=='Burgers':
        model = FNO1d()
    elif pde_name=='Darcy_Flow':
        model = FNO2d()
    elif pde_name=='Navier_Stokes_2D':
        model = FNO3d()
            
    return model

def train(model, pde_name, train_loader, loss_fn, optimizer, scheduler, device):  
    model.train()
    train_loss = 0.0
    train_data=0.0
    train_f = 0.0
    data_weight = 10
    f_weight = 0.01
    ic_weight= 0.01
    
    # train loop
    for x,y,grid in train_loader:
        x, y, grid = to_device([x,y,grid],device)

        batch_size= x.shape[0]
        
        y_pred= model(x,grid)
        data_loss = loss_fn(y_pred.reshape([batch_size,-1]),y.reshape([batch_size,-1]))

        if pde_name=='Burgers':
            dx=1/grid.shape[1]
            f_loss=burger_1d_loss(y_pred,x, y, dx)
            loss = data_weight*data_loss +f_weight*f_loss+ic_weight*ic_loss
        elif pde_name=='Darcy_Flow':
            dx=1/grid.shape[1]
            f_loss=darcy_loss(y_pred, x, y, dx)
            loss = data_weight*data_loss +f_weight*f_loss
        elif pde_name=='Navier_Stokes_2D':
            dx=1/grid.shape[1]
            dt=1/grid.shape[2]  
            ic_loss,f_loss=ns_2d_loss(y_pred, x, y, dx, dt)
            loss = data_weight*data_loss +f_weight*f_loss+ic_weight*ic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss+=loss.item()
        train_data+=data_loss.item()
        train_f+=f_loss.item()

    
    train_loss/=len(train_loader)
    train_data/=len(train_loader)
    train_f/=len(train_loader)

    return train_loss, train_data, train_f

def test(model, test_loader, loss_fn, device):  
    model.eval()  
    test_loss = 0.0  
    with torch.no_grad():  
        for x,y,grid in test_loader:  
            x=x.to(device)
            y=y.to(device)  
            grid=grid.to(device)
            batch_size= x.shape[0]
            y_pred = model(x,grid)    
            loss = loss_fn(y_pred.reshape([batch_size,-1]),y.reshape([batch_size,-1]))  
            test_loss += loss.item()
            
    return test_loss / len(test_loader) 


def main(args):
    # init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # checkpoint = torch.load(args["model_path"]) if not args["if_training"] or args["continue_training"] else None
    # saved_model_name = args['train'].get('save_name',args['model_name'])
    # saved_model_name = saved_model_name+ f"_lr{args['optimizer']['lr']}" + f"_bs{args['dataloader']['batch_size']}"
    # saved_dir = os.path.join(args["output_dir"], os.path.splitext(args["dataset"]["file_name"])[0])
    # if not os.path.exists(saved_dir):
    #     os.makedirs(saved_dir)


   
    model_name='PINO'
    
    pde_name=args.pde_name
    
    
    ntrain = 900
    ntest = 100

    lr=0.001
    epoch = 500
    batch_size=32


    train_loader,test_loader=get_dataloader(pde_name, ntrain,ntest,batch_size)

    
    model=get_model(pde_name)
   
    model.to(device)
    total_params=count_params(model)
    print(f"Total parameters: {total_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epoch,
                                                    steps_per_epoch=len(train_loader))
    
    loss_fn=LpLoss(size_average=True)
    train_history=[]
    test_history=[]
    start_epoch = 0
    min_val_loss = torch.inf
    
    # train
    print(f"start training from epoch {start_epoch}")
    pbar=tqdm(range(start_epoch,epoch), dynamic_ncols=True, smoothing=0.05)
    for epoch in pbar:
        ## train loop
        train_loss, train_time = train(model, pde_name, train_loader, loss_fn, optimizer, scheduler, device)
        pbar.set_description(f"[Epoch {epoch}] train_loss: {train_loss:.5e}, train_time: {train_time:.5e}")
        train_history.append(train_loss)
        
        if (epoch+1) % 20 == 0:
            test_loss= test(model, test_loader, loss_fn, device)
            test_history.append(test_loss)
            print(f"[Epoch {epoch}] test_loss: {test_loss:.5e}")
            print("================================================",flush=True)
            if test_loss < min_val_loss:
                ### save best
                min_val_loss=test_loss
                torch.save(model.state_dict(), f"{model_name}_{pde_name}.pt")
            
    np.save(f'train_history_{model_name}_{pde_name}.npy',train_history)
    np.save(f'test_history_{model_name}_{pde_name}.npy',test_history)
    
    with open('train_result.txt','+a') as f:
        f.write(f'{model_name} {pde_name} test_loss: {min_val_loss}\n')
        f.write(f'Number of parameters: {total_params}\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINO")
    parser.add_argument("--pde_name", type=str, default="Navier_Stokes_2D", help="PDE name")
   
    args= parser.parse_args()
    main(args)
