# !/bin/bash

$EXISTING_DATASET=FALSE

if $EXISTING_DATASET=FALSE;
then

###############################################################
# Adversarially train a robust classifier with valina AT
# For Demo, we use l2 PGD attack and epislon=0.5
# The robustified model is saved at ./Robust/robust_resnet50.pt
################################################################

cd Robust

python adversarial_training.py --norm l2 --epsilon 0.5 --save robust_resnet50.pt

cd ..

wait

###############################################################
# Standardly train a non-robust classfier
# The non-robust model is saved at ./SL/non_robust_resnet50.pt
###############################################################

cd SL

CUDA_VISIBLE_DEVICES=5 python train_sl.py --save non_robust_resnet50.pt

cd ..

wait

###############################################################
# Generate the robust and non-robust version of CIFAR10
# The robust cifar10 is saved at ./data/robust_cifar10.pt
# The non-robust cifar10 is saved at ./data/non_robust_cifar10.pt
###############################################################

# If you use the robust ResNet50 checkpoint downloaded via our link, set normalize to True
CUDA_VISIBLE_DEVICES=5 python generate_dataset.py --pgd_steps 1000 --weight ./Robust/robust_resnet50.pt \
    --save ./data/robust_cifar10.pt --normalize --step_size 0.01 &

CUDA_VISIBLE_DEVICES=7 python generate_dataset.py --pgd_steps 1000 --weight ./SL/non_robust_resnet50.pt \
    --save ./data/non_robust_cifar10.pt --delta 2.0 --step_size 0.01 &

wait 

fi

############################################################
# Evaluate on supervised learning
# Results will be saved to ./SL/logs
############################################################

cd SL

# Clean CIFAR10
CUDA_VISIBLE_DEVICES=0 python train_sl.py --dataset cifar10 --model resnet18 &

# Robust CIFAR10
CUDA_VISIBLE_DEVICES=0 python train_sl.py --dataset robust-cifar10 --model resnet18 &

# Non-Robust CIFAR10
CUDA_VISIBLE_DEVICES=3 python train_sl.py --dataset non-robust-cifar10 --model resnet18 &

cd ..
wait

#############################################################
# Evaluate on contrastive learning
# Results will be saved to ./CL/logs
#############################################################

cd CL

# Clean CIFAR10
CUDA_VISIBLE_DEVICES=0 python train_cl.py --dataset cifar10 

# Robust CIFAR10
CUDA_VISIBLE_DEVICES=0 python train_cl.py --dataset robust-cifar10  

# Non-Robust CIFAR10
CUDA_VISIBLE_DEVICES=0 python train_cl.py --dataset non-robust-cifar10  

cd ..


#############################################################
# Evaluate on Masked Image Modeling
# Results will be saved to ./MIM/logs
#############################################################

cd MIM

# Clean CIFAR10
CUDA_VISIBLE_DEVICES=5 python train_mim.py --dataset cifar10 

# Robust CIFAR10
CUDA_VISIBLE_DEVICES=6 python train_mim.py --dataset robust-cifar10 

# Non-Robust CIFAR10
CUDA_VISIBLE_DEVICES=0 python train_mim.py --dataset non-robust-cifar10 

cd ..

wait

#############################################################
# Evaluate on Diffusion Model
# Results will be saved to ./DM/logs
#############################################################

cd DM

# Clean CIFAR10
python train_dm.py --dataset cifar10
               
# Robust CIFAR10
python train_dm.py --dataset robust-cifar10 

# Non-Robust CIFAR10
python train_dm.py --dataset non-robust-cifar10 

cd ..


