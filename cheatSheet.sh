module load v100
# request an interactive session, which allows the package installer to see the GPU hardware
srun --pty -n 1 -c 2 --time=00:10:00 --gres=gpu:1 --mem=16G bash -l runJob.sh
srun --pty -n 1 -c 2 --time=02:00:00 --mem=300G bash -l runJob.sh

conda activate multimedbench

# (optional) confirm the gpu is available. The output should show basic information about at least
# one GPU.
watch -n0.1 nvidia-smi

# Run the benchmark script
python3 benchmark.py

# To stop the job
exit

# To view current jobs
squeue -u croyer

# To connect to a running job
srun --pty --overlap --jobid ID bash
