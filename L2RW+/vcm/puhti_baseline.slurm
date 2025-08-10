#!/bin/bash
#SBATCH --job-name=L2RW_vcm
#SBATCH --account=project_2005312
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --time=24:00:00
#SBATCH --output=puhti_logs/baseline_ca_run.out

# 获取时间戳，并创建日志子目录
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logdir=puhti_logs/$timestamp
mkdir -p "$logdir"

# 保存当前脚本副本
cp "$0" "$logdir/puhti_$timestamp.slurm"

# 加载环境
module load pytorch

# 进入项目路径
cd /scratch/project_2005312/yan/ReID/L2RW+/Extend/video/vcm/
export OMP_NUM_THREADS=4

# 记录开始时间
start_time=$(date +%s)
# 所有输出重定向到 mask1k.log
{
echo "===== Running vcm====="
python train_baseline.py \
    --lr 0.2 \
    --seq_lenth 6 \
    --batch-size 8 \
    --max_epoch 100 \
    --num_pos 4 \
    --test-batch 32 \
    --seed 0
echo "===== Finished vcm training ====="

# 记录结束时间
end_time=$(date +%s)

# 计算总耗时
runtime=$((end_time - start_time))
minutes=$((runtime / 60))
seconds=$((runtime % 60))

echo "=============================================================="
echo "Total running time: ${minutes} minutes ${seconds} seconds"

} | tee "$logdir/vcm.log"