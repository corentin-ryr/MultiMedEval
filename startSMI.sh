ID=$(squeue -u $USER | tail -1| awk '{print $1}')

echo $ID

srun --pty --overlap --jobid $ID bash
