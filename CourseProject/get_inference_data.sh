object=Arrow 
rm -rf ./PDM_dataset_inference/"$object"/*

total_env=100
chunk_env=100

chunk_num=$total_env/$chunk_env
for ((i=0;i<chunk_num;i++))
do
    echo "chunk idx " $i 
    python get_inference_data.py --object $object --num_envs $chunk_env --chunk_idx $i
done

python CFVS/depth2pcd.py --visualize --data my --mode inference --type '' --object "$object"