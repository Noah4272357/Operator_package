# coding=utf-8
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from timeit import default_timer
from deeponet import *
from utils import *
from tqdm import tqdm

def get_model(pde_name,input_size,dual_path=0):
    if dual_path==1:
        if pde_name=='Burgers':
            model = DPDON1D(size=input_size,
                        query_dim= 1)
        elif pde_name=='Darcy_Flow':
            model = DPDON2D(size=input_size,
                        query_dim= 2)
        elif pde_name=='Navier_Stokes_2D':
            model = DPDON3D(size=64,
                        query_dim= 3,
                        time_step=10)
    elif dual_path==0:
        if pde_name=='Burgers':
            model = DeepONet1D(size=input_size,
                        query_dim= 1)
        elif pde_name=='Darcy_Flow':
            model = DeepONet2D(size=input_size,
                        query_dim= 2)
        elif pde_name=='Navier_Stokes_2D':
            model = DeepONet3D(size=64,
                        query_dim= 3,
                        time_step=10)
            
    elif dual_path==-1:
        if pde_name=='Burgers':
            model = DenseDON1D(size=input_size,
                        query_dim= 1)
        elif pde_name=='Darcy_Flow':
            model = DenseDON2D(size=input_size,
                        query_dim= 2)
        elif pde_name=='Navier_Stokes_2D':
            model = DenseDON3D(size=64,
                        query_dim= 3,
                        time_step=10)        
    return model
    
    



def train(model, train_loader, loss_fn, optimizer, scheduler, device):  
    t1=default_timer()
    model.train()
    train_loss = 0.0  
    for x,y,grid in train_loader:  
        batch_size = x.size(0)
        x=x.to(device)
        y=y.to(device)  
        grid=grid.to(device)
        
        optimizer.zero_grad()  
        y_pred = model(x,grid)    
        loss = loss_fn(y_pred.reshape([batch_size,-1]),y.reshape([batch_size,-1]))  
        loss.backward()  
        optimizer.step()  
        
        train_loss += loss.item()  
        scheduler.step()
    t2=default_timer()
    return train_loss / len(train_loader),t2-t1


def test(model, test_loader, loss_fn, device):  
    model.eval()  
    test_loss = 0.0  
    with torch.no_grad():  
        for x,y,grid in test_loader: 
            batch_size = x.size(0) 
            x=x.to(device)
            y=y.to(device)  
            grid=grid.to(device)
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


    if args.dual_path==1:
        model_name='DPDON'
    elif args.dual_path==0:
        model_name='DeepONet'
    elif args.dual_path==-1:
        model_name='DenseDON'
    
    print(f"Model name: {model_name}")
    pde_name=args.pde_name
    
    
    ntrain = 900
    ntest = 100

    lr=0.001
    epoch = args.epoch
    batch_size=32


    train_loader,test_loader=get_dataloader(pde_name, ntrain,ntest,batch_size)

    if pde_name=='Darcy_Flow':
        input_size=64
    else:
        input_size=128
    
    model=get_model(pde_name,input_size,dual_path=args.dual_path)
   
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
        train_loss, train_time = train(model, train_loader, loss_fn, optimizer, scheduler, device)
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
    parser = argparse.ArgumentParser(description="DeepONet")
    parser.add_argument("--dual_path", type=int, default=1, help="Model type")
    parser.add_argument("--pde_name", type=str, default="Burgers", help="PDE name")
    parser.add_argument("--epoch", type=int, default=500)
   
    args= parser.parse_args()
    main(args)
