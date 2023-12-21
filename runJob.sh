# module load v100-32g
# srun --pty -n 1 -c 2 --time=00:10:00 --gres=gpu:1 --mem=64G bash -l runJob.sh 2>&1 | tee terminal.txt
# srun --pty -n 1 -c 1 --time=00:10:00 --mem=16G bash -l runJob.sh 2>&1 | tee terminal.txt


source ~/data/miniconda3/etc/profile.d/conda.sh

conda activate multimedbench

nvidia-smi

# (optional) confirm the gpu is available. The output should show basic information about at least
# one GPU.
# watch -n0.1 nvidia-smi

# Run the benchmark script
# python benchmark.py
# python testMIMIC.py
# python testImageClassification.py
# python MIMIC_entity_extraction.py
python testMedQA.py
# python testMIMIC3.py



# # To view current jobs
# squeue -u croyer

# # To connect to a running job
# srun --pty --overlap --jobid ID bash
