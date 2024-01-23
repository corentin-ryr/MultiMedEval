# module load v100-32g
# srun --pty -n 1 -c 8 --time=02:00:00 --gres=gpu:1 --mem=32G bash -l runJob.sh 2>&1 | tee terminal.txt
# srun --pty -n 1 -c 1 --time=00:10:00 --mem=16G bash -l runJob.sh 2>&1 | tee terminal.txt


source ~/data/miniconda3/etc/profile.d/conda.sh

conda activate multimedbench

free
nvidia-smi

# Run the benchmark script
# python benchmark.py
python testTasks.py
# python MIMIC_entity_extraction.py





# # To view current jobs
# squeue -u croyer

# # To connect to a running job
# srun --pty --overlap --jobid ID bash
