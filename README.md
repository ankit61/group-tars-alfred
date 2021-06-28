# TARS
This is the repo for the 11-777 class project at CMU.

### Project Abstract

For an artificially intelligent agent, performing everyday tasks given natural language instructions and egocentric visual input is very difficult. Recently, ALFRED, an instruction following and navigation dataset for household tasks, was proposed to advance the state of the art in embodied intelligence. Agents in ALFRED have to complete long horizon tasks with non-reversible actions, understand complex interactions between vision, language, and their own actions, and generalize to unseen objects and environments. While several models have been proposed to solve this challenging benchmark, all of them perform very poorly on both seen and unseen validation/test splits compared to humans. Through both quantitative and qualitative assessment, we argue that even the best models are still highly primitive. We note that an important shortcoming of all previous models is that they hope that incredibly rich and sophisticated representations will automatically emerge in their models without designing/enforcing them. To this end, we propose a novel model that can disentangle between explicit and implicit forms of representations and context, thereby significantly reducing the learning load of each component. While we do not have all final results yet, we show that our model is capable of fitting the training dataset significantly faster while using much fewer parameters than all previous models. Given more time, we are hopeful that we would be able to achieve strong results with our model.

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
