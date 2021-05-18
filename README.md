# TARS
This is the repo for the 11-777 class project at CMU.

### Instructions to run

In a terminal, run

```
pip3 install -r requirements.txt
python3 setup.py develop
# download ALFRED data

# run augment trajectories script
python3 tars/alfred/gen/scripts/augment_trajectories.py --split_type train --num_threads [NUM_THREADS]
python3 tars/alfred/gen/scripts/augment_trajectories.py --split_type valid_seen --num_threads [NUM_THREADS]
python3 tars/alfred/gen/scripts/augment_trajectories.py --split_type valid_unseen --num_threads [NUM_THREADS]

# start training the model
python3 tars/scripts/train.py --model TarsPolicy --dataset-splits_file tars/alfred/data/splits/oct21.json
```

### Acknowledgements

- Original ALFRED Repo: https://github.com/askforalfred/alfred
- MOCA Repo: https://github.com/gistvision/moca/
