#!/bin/bash

# 进程ID列表
pids=(31887)

# 运行持续时间（单位：秒）
duration=(28800) #28800 # 8 hours * 3600 seconds/hour
# 当前时间
start_time=$(date +%s)

# CPU使用率限制百分比
cpu_limit=90

# 遍历进程ID列表，对每个进程应用cpulimit
for pid in "${pids[@]}"; do
    echo "Limiting CPU usage of process $pid to $cpu_limit%"
    sudo cpulimit -l $cpu_limit -p $pid &
done

while true; do
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))

    if ((elapsed_time >= duration)); then
        echo "Stopping CPU usage limits after $((duration / 3600)) hours"
        sudo pkill cpulimit
        break
    fi

    # 每隔10秒检查一次
    sleep 10
done
