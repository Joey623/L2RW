#!/bin/bash
#SBATCH --job-name=L2RW_bupt
#SBATCH --account=project_2005312
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1,nvme:600
#SBATCH --time=24:00:00
#SBATCH --output=puhti_logs/l2rw_b_%j.out

# 获取时间戳，并创建日志子目录
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
logdir=puhti_logs/$timestamp
mkdir -p "$logdir"

# 保存当前脚本副本
cp "$0" "$logdir/puhti_$timestamp.slurm"

# 加载环境
module load pytorch
#cd /scratch/project_2005312/yan/ReID/L2RW+/dataset/
cp /scratch/project_2005312/yan/ReID/L2RW+/dataset/BUPTCampus.zip $LOCAL_SCRATCH
wait
cd $LOCAL_SCRATCH
unzip BUPTCampus.zip -d ./
wait

# 进入项目路径
cd /scratch/project_2005312/yan/ReID/L2RW+/Extend/video/bupt/
export OMP_NUM_THREADS=4

# 记录开始时间
start_time=$(date +%s)
# 所有输出重定向到 mask1k.log
{
echo "===== Running bupt====="
python train_baseline.py \
    --lr 0.2 \
    --train_bs 16 \
    --max_epoch 100 \
    --data_root $LOCAL_SCRATCH/DATA\
    --sequence_length 10 \
    --img_hw 256 128 \



echo "===== Finished bupt training ====="

# 记录结束时间
end_time=$(date +%s)

# 计算总耗时
runtime=$((end_time - start_time))
minutes=$((runtime / 60))
seconds=$((runtime % 60))

echo "=============================================================="
echo "Total running time: ${minutes} minutes ${seconds} seconds"

} | tee "$logdir/bupt.log"