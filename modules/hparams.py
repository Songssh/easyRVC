import os
import yaml

CFG_PATH = 'data/config.yaml'

def update_hparams(data):
    global CFG_PATH
    with open(CFG_PATH, 'w') as file:
        yaml.safe_dump(data, file)

def load_hparams(file_path):
    with open(file_path, 'r') as file:
        #return yaml.load(file, Loader=yaml.FullLoader)
        return yaml.safe_load(file)

hparams = load_hparams(CFG_PATH)
