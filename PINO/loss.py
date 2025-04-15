# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from utils import LpLoss

def burger_1d_loss(pred, a, u, dx, dt):  #
    # pred,u: [bs,x,t,1]
    # ic and f loss only, because bc can be infered from data
    v_coeff= 0.001
    loss_fn = nn.MSELoss(reduction="mean")
    #f_loss
    # pred: [bs,x,t,1]
    du_t=(pred[:,:,2:]-pred[:,:,:-2])/(2*dt)
    du_x=(pred[:,2:]-pred[:,:-2])/(2*dx)
    du_xx=(pred[:, 2:] -2*pred[:,1:-1] +pred[:, :-2]) / (dx**2)
    Df = du_t[:,1:-1] + du_x[:,:,1:-1]*pred[:,1:-1,1:-1] - v_coeff/np.pi*du_xx[:,:,1:-1]
    f_loss=loss_fn(Df, torch.zeros_like(Df))
    return f_loss
    

def darcy_loss(pred, a, u, dx):
    beta=1.0
    a=a.squeeze(-1)  # [bs, x, x, 1]
    du_x_= (pred[:,1:]-pred[:,:-1])/dx   # 1/2 step point
    du_y_= (pred[:,:,1:]-pred[:,:,:-1])/dx   # 1/2 step point
    ax_=(a[:,:-1]+a[:,1:])/2       # interpolating

    ay_=(a[:,:,1:]+a[:,:,:-1])/2
    Df= - ((ax_*du_x_)[:,1:,1:-1]-(ax_*du_x_)[:,:-1,1:-1]+(ay_*du_y_)[:,1:-1,1:]-(ay_*du_y_)[:,1:-1,:-1])/dx-beta
    f_loss= Df.norm(p=2)
    return f_loss  


def ns_2d_loss(pred, a, u, dx, dt):
    eta=0.1
    zeta=0.1
    gamma=5.0/3.0
    h = pred[..., 0:1]  # rho
    p = pred[..., 1:2]  # p
    u1 = pred[..., 2:3]  # vx
    u2 = pred[..., 3:4]  # vy
    E = p/(gamma - 1.) + 0.5 * h * (u1**2 + u2**2)
    Fx = u1 * (E + p)
    Fx = Fx
    Fy = u2 * (E + p)
    Fy = Fy
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss = loss_fn(h[...,0,:],u[...,0,0:1])+loss_fn(u1[...,0,:],u[...,0,1:2])+\
         loss_fn(u2[...,0,:],u[...,0,2:3])+loss_fn(p[...,0,:],u[...,0,3:4])
    # non conservative form
    dhu1_x = ((h*u1)[:,2:]-(h*u1)[:,:-2])/(2*dx)
    dhu2_y = ((h*u2)[:,:,2:]-(h*u2)[:,:,:-2])/(2*dx)
    dh_t = (h[:,:,:,2:]-h[:,:,:,:-2])/(2*dt)
    du1_x = (u1[:,2:]-u1[:,:-2])/(2*dx)
    du1_y = (u1[:,:,2:]-u1[:,:,:-2])/(2*dx)
    du1_t = (u1[:,:,:,2:]-u1[:,:,:,:-2])/(2*dt)
    du2_x = (u2[:,2:]-u2[:,:-2])/(2*dx)
    du2_y = (u2[:,:,2:]-u2[:,:,:-2])/(2*dx)
    du2_t = (u2[:,:,:,2:]-u2[:,:,:,:-2])/(2*dt)
    dp_x = (p[:,2:]-p[:,:-2])/(2*dx)
    dp_y = (p[:,:,2:]-p[:,:,:-2])/(2*dx)
    dFx_x = (Fx[:,2:]-Fx[:,:-2])/(2*dx)
    dFy_y = (Fy[:,:,2:]-Fy[:,:,:-2])/(2*dx)
    dE_t = (E[:,:,:,2:]-E[:,:,:,:-2])/(2*dt)

    du1_xx= (u1[:,2:]-2*u1[:,1:-1]+u1[:,:-2])/(dx**2)
    du1_yy= (u1[:,:,2:]-2*u1[:,:,1:-1]+u1[:,:,:-2])/(dx**2)
    du1_xy= (du1_x[:,:,2:]-du1_x[:,:,:-2])/dx
    du2_xx= (u2[:,2:]-2*u2[:,1:-1]+u2[:,:-2])/(dx**2)
    du2_yy= (u2[:,:,2:]-2*u2[:,:,1:-1]+u2[:,:,:-2])/(dx**2)
    du2_xy= (du2_x[:,:,2:]-du2_x[:,:,:-2])/dx
    eq1 = dh_t[:,1:-1,1:-1] + dhu1_x[:,:,1:-1,1:-1] + dhu2_y[:,1:-1,:,1:-1]
    eq2 = h[:,1:-1,1:-1,1:-1] * (du1_t[:,1:-1,1:-1] + u1[:,1:-1,1:-1,1:-1] * du1_x[:,:,1:-1,1:-1] + u2[:,1:-1,1:-1,1:-1] * du1_y[:,1:-1,:,1:-1]) + dp_x[:,:,1:-1,1:-1] - eta*(du1_xx[:,:,1:-1,1:-1]+du1_yy[:,1:-1,:,1:-1])-(zeta+eta/3.0)*(du1_xx[:,:,1:-1,1:-1]+du2_xy[:,:,:,1:-1])
    eq3 = h[:,1:-1,1:-1,1:-1] * (du2_t[:,1:-1,1:-1] + u1[:,1:-1,1:-1,1:-1] * du2_x[:,:,1:-1,1:-1] + u2[:,1:-1,1:-1,1:-1] * du2_y[:,1:-1,:,1:-1]) + dp_y[:,1:-1,:,1:-1] - eta*(du2_xx[:,:,1:-1,1:-1]+du2_yy[:,1:-1,:,1:-1])-(zeta+eta/3.0)*(du1_xy[:,:,:,1:-1]+du2_yy[:,1:-1,:,1:-1])
    eq4 = dE_t[:,1:-1,1:-1] + dFx_x[:,:,1:-1,1:-1] + dFy_y[:,1:-1,:,1:-1]

    f_loss= loss_fn(eq1,torch.zeros_like(eq1))+loss_fn(eq2,torch.zeros_like(eq2))+\
         loss_fn(eq3,torch.zeros_like(eq3))+loss_fn(eq4,torch.zeros_like(eq4))
    return ic_loss, f_loss
 

def burger_2d_loss(pred, a, u, dx, dt):
    v_coeff= 0.001
    loss_fn = nn.MSELoss(reduction="mean")
    ic_loss=loss_fn(pred[:,:,:,0],u[:,:,:,0])
    
    u1=pred[...,0:1]
    u2=pred[...,1:2]
    du1_t=(u1[:,:,:,2:]-u1[:,:,:,:-2])/(2*dt)
    du2_t=(u2[:,:,:,2:]-u2[:,:,:,:-2])/(2*dt)
    du1_x=(u1[:,2:]-u1[:,:-2])/(2*dx)
    du1_xx=(u1[:, 2:] -2*u1[:,1:-1] +u1[:, :-2]) / (dx**2)
    du1_y=(u1[:,:,2:]-u1[:,:,:-2])/(2*dx)
    du1_yy=(u1[:,:,2:] -2*u1[:,:,1:-1] +u1[:,:,:-2]) / (dx**2)
    du2_x=(u2[:,2:]-u2[:,:-2])/(2*dx)
    du2_xx=(u2[:, 2:] -2*u2[:,1:-1] +u2[:, :-2]) / (dx**2)
    du2_y=(u2[:,:,2:]-u2[:,:,:-2])/(2*dx)
    du2_yy=(u2[:,:,2:] -2*u2[:,:,1:-1] +u2[:,:,:-2]) / (dx**2)
    eq1= du1_t[:,1:-1,1:-1]+ u1[:,1:-1,1:-1,1:-1]*du1_x[:,:,1:-1,1:-1]+u2[:,1:-1,1:-1,1:-1]*du1_y[:,1:-1,:,1:-1]-v_coeff*(du1_xx[:,:,1:-1,1:-1]+du1_yy[:,1:-1,:,1:-1])
    eq2= du2_t[:,1:-1,1:-1]+ u1[:,1:-1,1:-1,1:-1]*du2_x[:,:,1:-1,1:-1]+u2[:,1:-1,1:-1,1:-1]*du2_y[:,1:-1,:,1:-1]-v_coeff*(du2_xx[:,:,1:-1,1:-1]+du2_yy[:,1:-1,:,1:-1])

    f_loss=loss_fn(eq1,torch.zeros_like(eq1))+loss_fn(eq2,torch.zeros_like(eq2))
    return ic_loss, f_loss


# lossmap={
#     '1D_Burgers':burger_1d_loss,
#     "1D_Advection":adv_1d_loss,
#     "1D_diffusion_sorption":diff_sorp_1d_loss,
#     "1D_diffusion_reaction":diff_react_1d_loss,
#     "1D_compressible_NS": CFD_1d_loss,
#     "2D_DarcyFlow":darcy_loss,
#     "2D_diffusion_reaction":diff_react_2d_loss,
#     "2D_shallow_water":swe_2d_loss,
#     "2D_Compressible_NS":CFD_2d_loss,
#     "1D_Allen_Cahn":Allen_Cahn_loss,
#     "1D_Cahn_Hilliard":Cahn_Hilliard_loss,
#     "2D_Burgers":burger_2d_loss
# }

# class Pdeloss:
#     def __init__(self, train_args):
#         super().__init__()
#         self.dx=train_args["dx"]
#         self.dt=train_args["dt"]
#         self.pdeloss=lossmap[train_args["scenario"]]
#     def __call__(self, pred, a, u):
#         return self.pdeloss(pred,a,u,self.dx,self.dt)
