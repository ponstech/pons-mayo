#!/bin/bash
NOW=$(date +"%Y-%m-%d/%H-%M-%S")

# OUT="hydra.run.dir=/media/GAIC_G13-1/users/Xiang/Thermal_AU_Logs/new/${NOW}"
OUT="hydra.run.dir=/data/Xiang/Thermal_AU_Logs/${NOW}"


HOSTNAME=`uname -n`

array=(4 5 6 7)
function function_panes() {
    FOLD_OUT=$OUT/"fold_"$1 #fold_1
    GPU=${array[$1-1]}  # 0, 5, 6
    # echo $GPU
    # echo $FOLD

    FOLD_CON="data_config.fold_num="$1
    HOST_CON="data_config.hostname="$HOSTNAME

    CMD="python train.py gpu_ids=$GPU $DATA_CON $FOLD_CON $HOST_CON $FOLD_OUT"

    # echo $CMD
    tmux send-keys -t $1 "$CMD" Enter
}





function_panes 1
function_panes 2
function_panes 3
function_panes 4
