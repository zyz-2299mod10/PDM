# pdm-f24-hw4
NYCU Perception and Decision Making 2024 Fall

In your original pdm-f24 directory, `git pull` to get new `hw4` directory.

## End-to-end Autonomous Driving Framework
For this homework, you will need to run the CARLA Simulator. Please ensure your system meets the following requirements:
- **System Requirements**: We recommend using Ubuntu 18.04 or Ubuntu 20.04.
- **An adequate GPU**: CARLA aims for realistic simulations, so the server needs at least 10 GB GPU although we would recommend 12 GB.
- **Disk space**: CARLA will use about 20 GB of space.
## Environment Setup
1. **Download the CARLA Package:**
    
    Download the **[Ubuntu] CARLA_0.9.15.tar.gz** Package   from the [CARLA GitHub releases page](https://github.com/carla-simulator/carla/releases), and extract it to the `pdm-f24/hw4/CARLA_0.9.15` directory.
2. **Download the Model Checkpoint and Testing Data for Coordinate Transformation:** 
    
    Download the model’s checkpoint and testing data from [Google Drive](https://drive.google.com/drive/folders/1EH0KRhf8-4f0X1h3rsOAB8FQmzI85ULn?usp=sharing).

    Place the checkpoint files in the `hw4/team_code/checkpoint` directory and the testing data in the `hw4/data1` directory.
3. **Set Up and Activate the Conda Environment:**

    ```shell
    conda env create -f environment.yml
    conda activate pdm-hw4
    ```

## Inference
1. **Run the inference script:**
    ```shell
    bash inference_stp3_by_autopilot.sh
    ```
2. **CARLA Server:**

    The CARLA server will launch, and after a default delay of 10 seconds, the test will automatically begin.
3.	**Results and Metrics:**

	•	The predicted future segmentation outputs will be stored in `logs/[Scenario Name]/debug_stp3`.

	•	Leaderboard metrics will be recorded in `/leaderboard/data/pdm_hw4.json`.

## Test your Affine Transformation
We have provided testing code for **TODO2_1** and **TODO2_2** in the `/test_for_future_motion` directory. The code loads two frames of the ego car, along with their coordinates and corresponding BEV maps.

You can execute the code by running: `python main.py`

After executing the code, three images will be generated in the folder:

- **mask_0_ori.png**: The original BEV of the first frame.
- **mask_1_ori.png**: The original BEV of the second frame. 
- **merge.png**: A merged image created by applying the affine transformation to the first frame. 

If your affine transformation is implemented correctly, the two BEV maps will align perfectly.

For details on performing the correct transformation, please refer to the [Documentation of Carla's coordinate system](https://github.com/autonomousvision/carla_garage/blob/main/docs/coordinate_systems.md)
## Notes

1.	**Locate TODOs:**

    Search for `# TODO_` in the code. There are five places where you need to implement. Ensure all are completed; otherwise, the inference script will not work.

2.	**Rerun Inference:**

    If you need to rerun the inference script, you need to delete the result file at `leaderboard/data/pdm_hw4.json`.
3.	**Handle Abnormal Exits:**

    If CARLA exits unexpectedly, you can clean the existing CARLA process by running: `bash clean_carla.sh`
4.	**QA page:**

    If you have any questions or issues, please post them on the [Notion page](https://lopsided-soursop-bec.notion.site/HW4-QA-sheet-0fe0e1834dbb4437b326b1a999cd10be?pvs=4).


## Reference

[Lift, Splat, Shoot - NVIDIA](https://research.nvidia.com/labs/toronto-ai/lift-splat-shoot/)

[ST-P3 - OpenDriveLab](https://arxiv.org/abs/2207.07601)
