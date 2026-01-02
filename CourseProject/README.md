# Abstract
In this work, we want to achieve the peg-insertion job with more complex peg and peghole by only using camera observation. we kept and retained the OAKN step of the CFVS process and utilized the traditional Corner Prediction algorithm to identify the keypoints of the peghole and the peg shape. This was done to determine the most suitable insertion pose. For this study, we used an arrow shape as the primary experimental configuration. 

# Method
<img width="1326" height="559" alt="image" src="https://github.com/user-attachments/assets/a060d726-a48e-42ed-ab40-6263a19bb946" />



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

cd ./CFVS/mankey
python ./train_pointnet2_kpts_no_oft.py

# After training complete, put the training result in the `log` to the `./hole_estimation/mankey/log/kpts`
```

# Demo
https://github.com/user-attachments/assets/6fa081ef-2962-4d92-9608-2d426d8e85af


https://github.com/user-attachments/assets/a099b4fd-6ba1-4998-9d00-3428757aa518

## Failure case

https://github.com/user-attachments/assets/c00e4da9-5c27-437f-b6a6-422fb5ab7818




