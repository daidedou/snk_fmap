import os

def faust_name(index, info_remeshed=None):
    if info_remeshed is None:
        return f"tr_reg_{index:03d}" + ".ply"
    elif info_remeshed:
        return f"R_tr_reg_{index:03d}" + ".ply"
    else:
        return f"O_tr_reg_{index:03d}" + ".ply"

def mesh_file_name(index, dataset, info_remeshed=None):
    filename = ""
    if dataset == "FAUST":
        filename = f"tr_reg_{index:03d}" + ".ply"
    elif dataset == "SCAPE": 
        filename = f"mesh{index:03d}" + ".ply"
    elif dataset == "SHREC":
        filename = f"{index}.ply"
    else:
        raise Exception(f"Uknown {dataset} dataset")
    if info_remeshed is None:
        return filename
    elif info_remeshed:
        return "R_" + filename
    else:
        return "O_" + filename
        

def get_exp_file(folder, dataset, index, file_type, ext, remeshed=True, id_step=29):
    filename = f'{id_step}{file_type}.{ext}'
    filename = os.path.join(mesh_file_name(index, dataset, remeshed), filename)
    
    return os.path.join(folder, filename)