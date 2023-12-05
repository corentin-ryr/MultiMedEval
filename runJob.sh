source ~/miniconda3/etc/profile.d/conda.sh

conda activate multimedbench

nvidia-smi

# (optional) confirm the gpu is available. The output should show basic information about at least
# one GPU.
# watch -n0.1 nvidia-smi

# Run the benchmark script
# python benchmark.py
python testMIMIC.py