import os
from cycle_calc import calc_breath_cycle
base_dir = '../data/icbhi_dataset/audio_test_data'

for f_name in os.listdir(base_dir):
    if f_name.endswith(".wav"):
        fpath = os.path.join(base_dir, f_name)
        print(f_name, " breath cycle: ", calc_breath_cycle(fpath))