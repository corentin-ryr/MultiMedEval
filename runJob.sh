
ssh croyer@cluster.s3it.uzh.ch

# Get the info about the cluster (what's inside)
sinfo


module load gpu
# request an interactive session, which allows the package installer to see the GPU hardware
srun --pty -n 1 -c 2 --time=01:00:00 --gres=gpu:1 --mem=8G bash -l


# (optional) confirm the gpu is available. The output should show basic information about at least
# one GPU.
nvidia-smi

# use mamba (drop-in replacement for conda)

module load mamba

# create a virtual environment named 'torch' and install packages

mamba create -n torch -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8 transformers

# activate the virtual environemnt

source activate torch

# confirm that the GPU is correctly detected

python -c 'import torch as t; print("is available: ", t.cuda.is_available()); print("device count: ", t.cuda.device_count()); print("current device: ", t.cuda.current_device()); print("cuda device: ", t.cuda.device(0)); print("cuda device name: ", t.cuda.get_device_name(0)); print("cuda version: ", t.version.cuda)'

# when finished with your test, exit the interactive cluster session

conda deactivate
exit