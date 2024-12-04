#!/usr/bin/env -S bash -e

THIS_DIRECTORY="$(dirname $(readlink -f $0))"
cd $THIS_DIRECTORY

# uv run python standup_zoo.py --env NaoStandup-v1 --algo ppo --conf-file hyperparameters/ppo.yml --track --wandb-project-name nao_standup_zoo
uv run python record_video.py --env NaoStandup-v1 --algo ppo 
