pde_names=(
    'Darcy_Flow'
    'Burgers'
    'Navier_Stokes_2D'
)

dual_path=(-1 0 1)

# 遍历每个层数并运行训练  
for pde in "${pde_names[@]}"; do  
    for mode in "${dual_path[@]}"; do
        echo "Start training $pde data..."  
        python train.py --pde_name $pde --dual_path $mode 
    done  
done  