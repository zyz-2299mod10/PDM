#! /bin/bash

while getopts ":o:t:c:" opt; do
    case "$opt" in
        o)
            object="$OPTARG"
            ;;
        t)
            total_env="$OPTARG"
            ;;
        c)
            chunk_env="$OPTARG"
            ;;        
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done

rm -rf ./PDM_dataset/"$object"_fine/*

chunk_num=$(($total_env / $chunk_env))
for ((i=0;i<chunk_num;i++))
do
    echo "chunk idx " $i 
    python get_fine_data.py --object $object --num_envs $chunk_env --chunk_idx $i
done

python CFVS/depth2pcd.py --visualize --data my --mode train --type fine --object "$object" --crop_pcd

