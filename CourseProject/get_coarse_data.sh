object=Arrow 
rm -rf ./PDM_dataset/"$object"_coarse/*

total_env=5000
chunk_env=200

chunk_num=$total_env/$chunk_env
for ((i=0;i<chunk_num;i++))
do
    echo "chunk idx " $i 
    python get_coarse_data.py --object $object --num_envs $chunk_env --chunk_idx $i
done

python CFVS/depth2pcd.py --visualize --data my --mode train --type coarse --object "$object"