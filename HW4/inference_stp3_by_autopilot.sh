#!/bin/bash

# This script starts PDM-Lite and the CARLA simulator on a local machine

export CARLA_ROOT=./CARLA_0.9.15
export WORK_DIR=./
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

# carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export REPETITIONS=1
export DEBUG_CHALLENGE=0


export PTH_ROUTE=${WORK_DIR}/leaderboard/data/pdm_hw4
# Function to handle errors
handle_error() {
  pkill Carla
  exit 1
}

# Set up trap to call handle_error on ERR signal
trap 'handle_error' ERR

# Start the carla server
export PORT=$((RANDOM % (40000 - 2000 + 1) + 2000)) # use a random port

sh ${CARLA_SERVER} -carla-streaming-port=0 -carla-rpc-port=${PORT} &
# sh ${CARLA_SERVER} -RenderOffScreen -carla-streaming-port=0 -carla-rpc-port=${PORT} -graphicsadapter=0 &

sleep 10 # on a fast computer this can be reduced to sth. like 6 seconds

echo 'Port' $PORT

export TEAM_AGENT=${WORK_DIR}/team_code/stp3_with_autopilot.py
export CHALLENGE_TRACK_CODENAME=MAP
export ROUTES=${PTH_ROUTE}.xml
export TM_PORT=$((PORT + 3))

export CHECKPOINT_ENDPOINT=${PTH_ROUTE}.json

# model weight 
export TEAM_CONFIG=./team_code/checkpoint/prediction.ckpt

export PTH_LOG='logs'
export RESUME=0
export DATAGEN=1
export SAVE_PATH='logs'
export TM_SEED=0

# Start the actual evaluation / data generation
python leaderboard/leaderboard/leaderboard_evaluator_local.py --port=${PORT} --traffic-manager-port=${TM_PORT} --routes=${ROUTES} --repetitions=${REPETITIONS} --track=${CHALLENGE_TRACK_CODENAME} --checkpoint=${CHECKPOINT_ENDPOINT} --agent=${TEAM_AGENT} --agent-config=${TEAM_CONFIG} --debug=0 --resume=${RESUME} --timeout=2000 --traffic-manager-seed=${TM_SEED}

# Kill the Carla server afterwards
# pkill Carla


# 2 

