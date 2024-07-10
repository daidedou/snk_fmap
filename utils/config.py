import yaml 
import os 
config_vars = yaml.safe_load(open("utils/config.yaml", "r"))

def get_path_scape(remeshed=None):
    assert (remeshed is None) or (isinstance(remeshed, bool)), "Has to be boolean or None"
    if remeshed is None:
        return os.path.join(config_vars["data_nicolas"], "SCAPE")
    elif remeshed:
        return os.path.join(config_vars["data_nicolas"], "SCAPE", "R")
    else:
        return os.path.join(config_vars["data_nicolas"], "SCAPE", "O")        

def get_path_shrec(remeshed=None):
    assert (remeshed is None) or (isinstance(remeshed, bool)), "Has to be boolean or None"
    if remeshed is None:
        return os.path.join(config_vars["data_nicolas"], "SHREC19")
    elif remeshed:
        return os.path.join(config_vars["data_nicolas"], "SHREC19", 'R')
    else:
        return os.path.join(config_vars["data_nicolas"], "SHREC19", "O")


def get_path_faust(remeshed=None):
    assert (remeshed is None) or (isinstance(remeshed, bool)), "Has to be boolean or None"
    if remeshed is None:
        return os.path.join(config_vars["data_nicolas"], "FAUST")
    elif remeshed:
        return os.path.join(config_vars["data_nicolas"], "FAUST", "R")
    else:
        return os.path.join(config_vars["data_nicolas"], "FAUST", "O")

def get_path_dt4d(remeshed=None):
    assert (remeshed is None) or (isinstance(remeshed, bool)), "Has to be boolean or None"
    if remeshed is None:
        return os.path.join(config_vars["data_nicolas"], "DT4D")
    elif remeshed:
        return os.path.join(config_vars["data_nicolas"], "DT4D", "R")
    else:
        return os.path.join(config_vars["data_nicolas"], "DT4D", "O")


def get_dataset_path(dataset, remeshed=None):
    if dataset.lower() == "scape":
        return get_path_scape(remeshed)
    if dataset.lower() == "shrec":
        return get_path_shrec(remeshed)
    if dataset.lower() == "faust":
        return get_path_faust(remeshed)
    if dataset.lower() == "dt4d":
        return get_path_shrec()
    else:
        raise Exception("Unknown dataset " + dataset)


def get_x_config():
    return not(config_vars["ssh"])

def get_template_path():
    return config_vars["template_path"]


if __name__ == "__main__":
    print(get_path_shrec(True))
    print(get_path_shrec())