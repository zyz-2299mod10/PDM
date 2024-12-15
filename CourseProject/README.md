
# Quick start
> [!IMPORTANT]
> * Install the **IsaacGym** first
> * Remember to change the urdf path and the isaacgym asset path at `line 36, 37` in `dummy_main_simulate.py`

```
python dummy_main_simulate.py --object Arrow 
```

# Train on your CAD model

put the urdf file in `PDM_urdf`
Then, 
```
./get_coarse_data.py
```

```
# change the dataset path in `CFVS/config` and training code

cd ./CFVS
python ./train_pointnet2_kpts_no_oft.py

# After training complete, put the training result in the `log` to the `./hole_estimation/mankey/log/kpts`
```
