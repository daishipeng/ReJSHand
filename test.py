
import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes


import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import json
import numpy as np
from tqdm import tqdm
from functools import lru_cache
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler

from cfg import _CONFIG
from hand_net import HandNet
from eval_datataset import HandMeshEvalDataset
from utils import get_log_model_dir
from matplotlib import pyplot as plt
import cv2
from infer_to_json import infer_single_json
from infer_to_json import verts2pcd
from infer_to_json import align_w_scale
from PIL import Image


shape_is_mano = None
#from models.losses import *
from kp_preprocess import *
from models.mano_torch import * 
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d

import cv2

from transforms import _get_3rd_point



NUM_BODY_JOINTS = 1
NUM_HAND_JOINTS = 15
NUM_JOINTS = NUM_BODY_JOINTS + NUM_HAND_JOINTS
NUM_SHAPES = 10

MANO_PARAMS_PATH = "/home/asrock/quanbao/daishipeng/simpleHand/models"

def load_mano_params_c():
    
    #mano_path = os.path.join(MANO_PARAMS_PATH, "MANO_LEFT_C.pkl")

    mano_path = os.path.join(MANO_PARAMS_PATH, "MANO_RIGHT_C.pkl")
    
    with open(mano_path, 'rb') as mano_file:
        model_data = pickle.load(mano_file)
    return model_data



#MANO_DATA_LEFT = load_mano_params_c(False)
MANO_DATA_RIGHT = load_mano_params_c()





device = torch.device("cpu")
@lru_cache(1)
def get_faces():
    faces = np.array(MANO_DATA_RIGHT["f"]).astype("int32")    
    return torch.from_numpy(faces)






def main():

    val_cfg = _CONFIG['VAL1']
    
    #img=cv2.imread('/home/robot/mahru/simple_hand/simpleHand-main/data/test/00000002.jpg')
    log_model_dir = get_log_model_dir(_CONFIG['NAME'])
    model_path = os.path.join(log_model_dir, '/home/asrock/quanbao/daishipeng/simpleHand/train_log/fastvit-FCB/epoch_200')
    
    print(model_path)
    model = HandNet(_CONFIG, pretrained=False)

    checkpoint = torch.load(open(model_path, "rb"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    model.cuda()
    
    
    bmk1 = val_cfg['BMK1']
    dataset = HandMeshEvalDataset(bmk1["json_dir"], val_cfg["IMAGE_SHAPE"], bmk1["scale_enlarge"])

    pred_uv_list, xyz_pred_list, verts_pred_list, xyz_gt_list, verts_gt_list = infer_single_json(val_cfg, bmk1, model, rot_angle=0)

    for pred_uv, pred_xyz, pred_vertices, gt_joints, gt_vertices, ori_info in zip(pred_uv_list, xyz_pred_list, verts_pred_list, xyz_gt_list, verts_gt_list, dataset.all_info):
        ori_info['pred_uv'] = pred_uv
        ori_info['pred_xyz'] = pred_xyz
        ori_info['pred_vertices'] = pred_vertices
        ori_info['xyz'] = gt_joints
        ori_info['vertices'] = gt_vertices
        ori_info['K']=dataset.all_info
    
    data=dataset[0]
    new_K = data["K"]
    img_processed=data["img"]
    uv=data["uv"]
    xyz=data["xyz"]
    trans_mat_2d=data["trans_matrix_2d"]
    trans_mat_3d=data["trans_matrix_3d"]

    faces=torch.tensor(get_faces())
    pred_vertices=torch.Tensor(pred_vertices)
 
    h, w = img_processed.shape[:2]
    uv_norm = uv.copy()
    uv_norm[:, 0] /= w   
    uv_norm[:, 1] /= h

    coord_valid = (uv_norm > 0).astype("float32") * (uv_norm < 1).astype("float32") # Nx2x21x2
    coord_valid = coord_valid[:, 0] * coord_valid[:, 1]

    valid_points = [pred_uv[i] for i in range(len(pred_uv)) if coord_valid[i]==1]        
    if len(valid_points) <= 1:
            valid_points = pred_uv

    #points = np.array(valid_points)
    points=torch.Tensor(valid_points)
    pred_uv=torch.tensor(pred_uv)
    pred_xyz=torch.Tensor(pred_xyz)    

    print(faces.shape)
    print(pred_uv.shape)
    print(pred_vertices.shape)
    print(pred_xyz.shape)
  
    gt_vertices=torch.Tensor(gt_vertices)
    verts_rgb = torch.ones_like(gt_vertices)[None]  # (1, V, 3)
    verts_rgb[:,:,:]=torch.tensor([1, 0.8, 0.7])
    def save_to_obj_file(vertices, faces, filename):
        with open(filename, 'w') as file:
        # Write vertices
            for vertex in vertices:
                file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write faces
            for face in faces:
                file.write(f"f {face[0]} {face[1]} {face[2]}\n")

    save_to_obj_file(pred_vertices, faces, '1935.obj')


    

if __name__ == "__main__":
    from fire import Fire
    Fire(main)
