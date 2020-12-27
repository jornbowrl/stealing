import os 

import json 
import logging 
import numpy as np 

import time 
from torch.utils.tensorboard import SummaryWriter
import torchvision

import yaml

try:
    from yaml import CSafeLoader as YamlSafeLoader, CSafeDumper as YamlSafeDumper
except ImportError:
    from yaml import SafeLoader as YamlSafeLoader, SafeDumper as YamlSafeDumper
ENCODING = "utf-8"


def write_yaml(config_path, data, overwrite=False):
    """
    Write dictionary data in yaml format.
    :param root: Directory name.
    :param file_name: Desired file name. Will automatically add .yaml extension if not given
    :param data: data to be dumped as yaml format
    :param overwrite: If True, will overwrite existing files
    """
#     if not exists(root):
#         raise MissingConfigException("Parent directory '%s' does not exist." % root)

    file_path =config_path# os.path.join(root, file_name)
    yaml_file_name = file_path if file_path.endswith(".yaml") else file_path + ".yaml"

    if os.path.exists(yaml_file_name) and not overwrite:
        raise Exception("Yaml file '%s' exists as '%s" % (file_path, yaml_file_name))

    try:
        #with codecs.open(yaml_file_name, mode="w", encoding=ENCODING) as yaml_file:
        with open(yaml_file_name, mode="w", encoding=ENCODING) as yaml_file:
            yaml.dump(
                data, yaml_file, default_flow_style=False, allow_unicode=True, Dumper=YamlSafeDumper
            )
    except Exception as e:
        raise e


def read_yaml(config_path):
    """
    Read data from yaml file and return as dictionary
    :param root: Directory name
    :param file_name: File name. Expects to have '.yaml' extension
    :return: Data in yaml file as dictionary
    """
#     if not exists(root):
#         raise MissingConfigException(
#             "Cannot read '%s'. Parent dir '%s' does not exist." % (file_name, root)
#         )

    file_path =config_path# os.path.join(root, file_name)
    if not os.path.exists(file_path):
        raise MissingConfigException("Yaml file '%s' does not exist." % file_path)
    try:
#         with codecs.open(file_path, mode="r", encoding=ENCODING) as yaml_file:
        with open(file_path, mode="r", encoding=ENCODING) as yaml_file:
            return yaml.load(yaml_file, Loader=YamlSafeLoader)
    except Exception as e:
        raise e


def Params(json_path,is_json=True):
    def dict2obj(d):
        if isinstance(d, list):
            d = [dict2obj(x) for x in d]
        if not isinstance(d, dict):
            return d
        class C(object):
            pass
        o = C()
        for k in d:
            o.__dict__[k] = dict2obj(d[k])
        return o

#     from collections import namedtuple
    if is_json:
        with open(json_path) as f :
            params = json.load(f)
    else:
        params = read_yaml(json_path)
    return dict2obj(params)
#     return namedtuple('Struct', params.keys())(*params.values())
    
class old_Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path,is_json=True):
        if is_json:
            with open(json_path) as f:
                params = json.load(f)
        else :
            params = read_yaml(json_path)
            
        self.__dict__.update(params)


    def save(self, json_path,is_json=True):
        if is_json:
            with open(json_path, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
        else :
            write_yaml(self.__dict__, json_path)
                
    def update(self, json_path,is_json=True):
        """Loads parameters from json file"""
        if is_json:
            with open(json_path) as f:
                params = json.load(f)
        else :
            params = read_yaml(json_path)
        self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)
        
def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)











    