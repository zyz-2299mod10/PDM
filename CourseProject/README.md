
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

# Demo
https://github.com/user-attachments/assets/6fa081ef-2962-4d92-9608-2d426d8e85af


https://github.com/user-attachments/assets/a099b4fd-6ba1-4998-9d00-3428757aa518

## Failure case

https://github.com/user-attachments/assets/c00e4da9-5c27-437f-b6a6-422fb5ab7818



