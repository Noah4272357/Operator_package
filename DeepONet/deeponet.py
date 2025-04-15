# coding=utf-8
import torch
import torch.nn as nn
from utils import _get_act, _get_initializer


class MLP(nn.Module):
    """Fully-connected neural network."""

    def __init__(self, in_channel, out_channel, layer_sizes, activation, kernel_initializer):
        super().__init__()
        self.activation = _get_act(activation)
        initializer = _get_initializer(kernel_initializer)
        initializer_zero = _get_initializer("zeros")

        layer_sizes = [in_channel] + layer_sizes + [out_channel]
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32
                )
            )
            initializer(self.linears[-1].weight)
            initializer_zero(self.linears[-1].bias)
        
    def forward(self, inputs):
        x = inputs
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x

class DeepONet1D(nn.Module):
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size :int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()
        in_channel_branch=in_channel_branch*size
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)

        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        grid=grid[0]  
        num_points=grid.shape[0]
        grid=grid.reshape([num_points,self.query_dim])#(num_point, query_dim)
        batchsize=x.shape[0]
        x = x.reshape([batchsize,-1])
        # Branch net to encode the input function
        
        x = self.branch(x)
        # Trunk net to encode the domain of the output function
        grid = self.activation_trunk(self.trunk(grid))
        x = x.reshape([batchsize,self.out_channel,-1])
        grid = grid.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, grid)
        # Add bias
        x += self.b
        return x

class DeepONet2D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)

        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        # Trunk net to encode the domain of the output function
        grid = self.activation_trunk(self.trunk(grid))
        
        x = x.reshape([batchsize,self.out_channel,-1])
        grid = grid.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, grid)
        # Add bias
        x += self.b

        return x

class DeepONet3D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        time_step:int,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2*time_step
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.branch = MLP(in_channel_branch,out_channel_branch,layer_sizes, activation_branch, kernel_initializer)
        self.trunk = MLP(in_channel_trunk,out_channel_trunk,layer_sizes, self.activation_trunk, kernel_initializer)

        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        # Trunk net to encode the domain of the output function
        grid = self.activation_trunk(self.trunk(grid))
        
        x = x.reshape([batchsize,self.out_channel,-1])
        grid = grid.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, grid)
        # Add bias
        x += self.b

        return x

class DenseDON1D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_layers: int = 3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        self.query_dim=query_dim
        self.out_channel = out_channel
        self.num_layers = num_layers

        in_channel_branch=in_channel_branch*size
        in_channel_trunk=query_dim
        out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        
        self.trunk_list = torch.nn.ModuleList()

        self.trunk_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_layers):
            self.trunk_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        grid=grid[0]  
        num_points=grid.shape[0]
        grid=grid.reshape([num_points,self.query_dim])#(num_point, query_dim)
        batchsize=x.shape[0]

        x = self.branch(x.reshape([batchsize,-1]))
        
        basis= self.trunk_list[0](grid)
        for i in range(1,self.num_layers):
            new_basis = self.trunk_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)

            
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)

        # Add bias
        x += self.b
        return x     

        
class DenseDON2D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_layers: int = 3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        self.out_channel = out_channel
        self.query_dim=query_dim
        layer_sizes = [128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.num_layers = num_layers
        self.trunk_list = torch.nn.ModuleList()

        self.trunk_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_layers):
            self.trunk_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
            

        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        
        basis= self.trunk_list[0](grid)
        for i in range(1,self.num_layers):
            new_basis = self.trunk_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)
        # Add bias
        x += self.b
        return x


class DenseDON3D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        time_step:int,
        in_channel_branch: int=1,
        out_channel: int=1,
        num_layers:int =3,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2*time_step
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.out_channel = out_channel
        self.query_dim=query_dim
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

        self.num_layers = num_layers
        self.branch_list = torch.nn.ModuleList()
        self.trunk_list = torch.nn.ModuleList()

        for i in range(num_layers):
            self.trunk_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        
            if i>0:
                self.branch_list.append(MLP(in_channel_branch*i, out_channel_branch, layer_sizes, activation_branch, kernel_initializer) )
            else:
                self.branch_list.append(MLP(in_channel_branch, out_channel_branch, layer_sizes, activation_branch, kernel_initializer) )
            

        
       
    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))
        
        basis= self.trunk_list[0](grid)
        for i in range(1,self.num_layers):
            new_basis = self.trunk_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)
        # Add bias
        x += self.b
        return x

class DPDON1D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_res: int = 4,
        num_dense: int=3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        self.query_dim=query_dim
        self.out_channel = out_channel
        self.num_res = num_res
        self.num_dense = num_dense

        in_channel_branch=in_channel_branch*size
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.param_list=torch.nn.ModuleList()
        self.res_list = torch.nn.ModuleList()
        self.dense_list = torch.nn.ModuleList()

        for i in range(num_res):
            if i==0:
                self.res_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            else:
                self.res_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            self.param_list.append(nn.Conv1d(out_channel_trunk,out_channel_trunk,1))

        self.dense_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_dense):
            self.dense_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_branch+out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        grid=grid[0]  
        num_points=grid.shape[0]
        grid=grid.reshape([num_points,self.query_dim])#(num_point, query_dim)
        batchsize=x.shape[0]

        x = self.branch(x.reshape([batchsize,-1]))


        
        basis= self.dense_list[0](grid)
        for i in range(1,self.num_dense):
            new_basis = self.dense_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)

        basis1=self.res_list[0](grid)
        for i in range(1,self.num_res):
            new_basis = self.res_list[i](basis1)
            new_basis=self.param_list[i](new_basis)
            basis1=basis1+new_basis
        
        basis=torch.cat((basis,basis1),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)

        # Add bias
        x += self.b
        return x     

        
class DPDON2D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        in_channel_branch: int=1,
        out_channel: int=1,
        activation: str = "gelu",
        num_res: int = 4,
        num_dense: int=3,
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        self.num_res = num_res
        self.num_dense=num_dense
        self.out_channel = out_channel
        self.query_dim=query_dim
        layer_sizes = [128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.res_list = torch.nn.ModuleList()
        self.dense_list = torch.nn.ModuleList()

        self.res_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_res):
            self.res_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )

        self.dense_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_dense):
            self.dense_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2
        
        self.branch=MLP(in_channel_branch,out_channel_branch+out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
        
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))


        
        basis= self.dense_list[0](grid)
        for i in range(1,self.num_dense):
            new_basis = self.dense_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)

        basis1=self.res_list[0](grid)
        for i in range(1,self.num_res):
            new_basis = self.res_list[i](basis1)
            basis1=basis1+new_basis
        
        basis=torch.cat((basis,basis1),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)
        # Add bias
        x += self.b
        return x


class DPDON3D(nn.Module): 
    #   For multiple outputs, we choose the second approach mentioned in "https://arxiv.org/abs/2111.05512", i.e. split 
    # the output of both the branch and the trunk into n groups, and the k-th groups outputs the k-th solution.
    def __init__(self,
        size: int,
        query_dim: int ,
        time_step:int,
        in_channel_branch: int=1,
        out_channel: int=1,
        num_res: int = 4,
        num_dense: int=3,
        activation: str = "gelu",
        kernel_initializer: str = "Glorot normal"):
        super().__init__()

        in_channel_branch=in_channel_branch*size**2*time_step
        in_channel_trunk=query_dim
        out_channel_branch=out_channel_trunk=128*out_channel
        layer_sizes=[128]*4
        
        activation_branch = self.activation_trunk = _get_act(activation)
        
        self.out_channel = out_channel
        self.query_dim=query_dim
        self.num_res = num_res
        self.num_dense=num_dense
        self.b = torch.nn.parameter.Parameter(torch.zeros(out_channel,dtype=torch.float32))

        self.res_list = torch.nn.ModuleList()
        self.dense_list = torch.nn.ModuleList()

        self.res_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_res):
            self.res_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )

        self.dense_list.append(MLP(in_channel_trunk, out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
        for i in range(1,num_dense):
            self.dense_list.append(MLP(out_channel_trunk,out_channel_trunk, layer_sizes, self.activation_trunk, kernel_initializer) )
            out_channel_trunk*=2

        self.branch=MLP(in_channel_branch,out_channel_branch+out_channel_trunk, layer_sizes, activation_branch, kernel_initializer)
        
        
       
    def forward(self, x, grid):
        batchsize=x.shape[0]
        # if grid are same, only take one 
        grid = grid[0]
        grid = grid.reshape([-1,self.query_dim])  #(num_point, query_dim)
        num_points=grid.shape[0]
        # Branch net to encode the input function
        x = self.branch(x.reshape([batchsize,-1]))


        
        basis= self.dense_list[0](grid)
        for i in range(1,self.num_dense):
            new_basis = self.dense_list[i](basis)
            basis=torch.cat((basis,new_basis),dim=1)

        basis1=self.res_list[0](grid)
        for i in range(1,self.num_res):
            new_basis = self.res_list[i](basis1)
            basis1=basis1+new_basis
        
        basis=torch.cat((basis,basis1),dim=1)
        
        x = x.reshape([batchsize,self.out_channel,-1])
        basis = basis.reshape([num_points,self.out_channel,-1])
        x = torch.einsum("bci,nci->bnc", x, basis)

        # Add bias
        x += self.b
        return x 
