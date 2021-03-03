import json
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Build table for generating dataset statistics
splits = ["train", "valid_seen", "valid_unseen"]
task_fields = ["task_type", "focus_object", "base_object", "dest_object", "scene"]
data_path = "../tars/alfred/data/json_2.1.0"

pseudo_attention_metric = [] # list of tuples (number of objects in ste-by-step instructions, total number of objects as per object_poses)
corresponding_files = []

for split in splits:
    try:
        task_dirs = os.listdir("{}/{}".format(data_path, split))
        for i in range(len(task_dirs)):
            task_dir = task_dirs[i]
            task_values = task_dir.split("-")
            for trial_dir in os.listdir("{}/{}/{}".format(data_path, split, task_dir)):
                traj_data_file = open("{}/{}/{}/{}/traj_data.json".format(data_path, split, task_dir, trial_dir))
                traj_data = json.load(traj_data_file)
                object_poses_json = traj_data['scene']['object_poses']
                object_name_list = []
                for object_pose_json in object_poses_json:
                    objectName = object_pose_json["objectName"]
                    object_name_list.append(objectName[:objectName.find("_")])

                object_name_list = list(set(object_name_list)) #remove duplicates

                # Get the step by step instructions
                count = 0
                words_already_accounted = []
                for instructions_json in traj_data["turk_annotations"]['anns']:
                    for step_by_step_instruction in instructions_json["high_descs"]:
                        for word in object_name_list:
                            if word.lower() not in words_already_accounted and word.lower() in step_by_step_instruction:
                                count += 1
                                words_already_accounted.append(word.lower())

                pseudo_attention_metric.append((count, len(object_name_list), count/len(object_name_list)))
                corresponding_files.append(traj_data_file)
                if pseudo_attention_metric[-1][-1] > 1:
                    print(traj_data_file)
    except:
        pass

pseudo_attention_metric = np.asarray(pseudo_attention_metric)
percentages = pseudo_attention_metric[:, 2]
plt.hist(percentages, bins=10)
plt.ylabel("Frequency")
plt.xlabel("%")
plt.title("Percentage of objects referred to in instructions \n compared to objects in the scene")
plt.show()