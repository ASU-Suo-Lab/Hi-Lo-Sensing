set -x

# partition='VC2' # 分区名
partition='VC3' # 分区名
# partition='INTERN2' # 分区名
# partition='INTERN3' # 分区名
TYPE='spot' # spot: 会被reserved挤占
# TYPE='reserved' # reserved: 可以一直运行
GPUS_PER_NODE=1 #单节点GPU数
GPUS=1 #总GPU数
CPUS_PER_TASK=12
JOB_NAME="wzk_debug"

# 计算节点开启代理联网

export http_proxy=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.51:23128/
export https_proxy=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.51:23128/
export HTTP_PROXY=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.51:23128/
export HTTPS_PROXY=http://liqingyun:z6pvK3s7BrOxj4BQQl6LvrgWzDSlIcVe4rQatSB5z4vsq4OSw6Qo1q59mpWa@10.1.20.51:23128/

    # -n${GPUS} --gres=gpu:${GPUS_PER_NODE} \
    # --ntasks-per-node=${GPUS_PER_NODE} \

srun -p $partition -w SH-IDCA1404-10-140-54-123 --job-name=${JOB_NAME} ${SRUN_ARGS} \
    --mpi=pmi2 --quotatype=${TYPE} \
    --cpus-per-task=$CPUS_PER_TASK \
    --kill-on-bad-exit=1 \
    python -u carla_train_eval.py --device=6 --fusion='raw' --log_name='raw_wo_velo' --remove_velocity --large 2>&1 | tee run_logs/raw_wo_velo_large.log