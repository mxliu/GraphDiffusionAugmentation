# Graph-Contrastive-Learning-with-Diffusion-Augmentation
This is the code for the "Self-Supervised Graph Contrastive Learning with Diffusion Augmentation" project

# Paper
Self-Supervised Graph Contrastive Learning with Diffusion Augmentation for Functional MRI Analysis and Brain Disorder Detection

Xiaochuan Wang, Yuqi Fang, Qianqian Wang, Pew-Thian Yap, Hongtu Zhu, Mingxia Liu

# Dependencies 
numpy=1.24.3
scipy=1.10.1
torch=2.1.2+cu118
einops=0.7.0
torchmetrics=1.1.1
wandb=0.15.10

# Usage
- `pretraining.py`: The pre-training of the proposed GCDA.
- `pretext_model.py`: The pretext model for pre-training mainly include graph diffusion augmentation (GDA) and graph contrastive learning.
- `diffusion_model.py`: The main functions for the GDA module mainly include noise unit and denoising neural network.
- `noisy_schedule.py`: The transition function in noise unit.
- `transformer_model.py`: This is denoising neural network.
- `GIN_encoder.py`: This is graph feature extraction backbone. 
- `diffusion_utils.py`: These are some useful functions in the GDA module. 
- `diffusion_loss`: This is diffusion loss function.
- `extra_features`: The calculating functions for global feature. 
- `extra_features1`: The copy of the extra_features for computing input dimensions of the denoising neural network. 
- `dataset`: Data preparation for pre-training. 
- `fine_tune`: The fine-tuning of the proposed GCDA.
- `fine_tune_model.py`: The task-specific model for fine-tuning.
- `dataset1`: Data preparation for fine-tuning. 

# Contact
If you have any problem with our code or have some suggestions, please feel free to contact us: 

- Xiaochuan Wang (xiaochuan10052022@163.com)
- Qianqian Wang (qqw@email.unc.edu)
- Mingxia Liu (mingxia_liu@med.unc.edu)
