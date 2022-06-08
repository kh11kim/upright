from asyncore import write
import numpy as np
from upright.utils.world import BulletWorld
from upright.utils.scene_maker import BulletSceneMaker
from upright.utils.body import Body
from upright.utils.robot import Panda
from upright.utils.transform import Transform, Rotation
from upright.utils.camera import CameraIntrinsic
import pybullet_data
from PIL import Image
import uuid
from pathlib import Path
from typing import Iterable

def watch_workspace(world, target_position: np.ndarray):
    world.physics_client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=np.pi/2,
        cameraPitch=-10,
        cameraTargetPosition=target_position
    )

def main(obj_name: str, rot0: np.ndarray, upright:np.ndarray, rot_type: str):
    """
    Args:
        obj_name (str): the object name from YCB dataset
        rot (np.ndarray): quaternion or rotation matrix
        rot_type (str): "qtn"(quaternion), "mat"(matrix)
    """
    if rot_type == "qtn":
        qtn0 = rot0
    if rot_type == "mat":
        qtn0 = Rotation.from_matrix(rot0).as_quat()
    
    obj_path = "ycb/"+ obj_name +"/google_16k"

    #create environment
    world = BulletWorld(gui=True)
    sm = BulletSceneMaker(world)
    world.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    grasp_frame = Transform(Rotation.from_euler("xyz", [0,np.pi/2, 0]), [0,0,0])
    print(grasp_frame.rotation.inv().as_quat())
    hand = world.load_urdf("hand", "data/urdfs/panda/panda_hand.urdf")
    
    base_to_tcp = hand.get_link_pose(2)
    hand.set_base_pose(grasp_frame * base_to_tcp.inverse())
    
    #set scene
    target_point = [0, 0, 0]
    watch_workspace(world, target_point)
    
    sm.view_frame(grasp_frame, "grasp")
    T_grasp_obj = Transform(Rotation.from_quat(qtn0), [0,0,0])
    obj = world.load_ycb(
        name="obj", 
        path=obj_path, 
        pose=grasp_frame*T_grasp_obj
    )
    T_upright = T_grasp_obj * Transform(Rotation.from_quat(upright), [0,0,0])
    obj.set_base_pose(grasp_frame*T_upright)
    #pred_upright = Rotation.from_quat(qtn) * Rotation.from_quat(rot_est).inv()
    
    input()
    
if __name__ == "__main__":
    obj_name = "004_sugar_box"
    # init rotation
    qtn0 = np.array(
        [0.02776734731663275,-0.8053691864011093,-0.28570081079591764,0.5186371513198492]
    )
    # estimated upright orientation
    upright = np.array(
        [-0.13076883060306502,-0.23931564659427754,4.1633363423443364e-17,-0.9620953872864528]
    )
    main(obj_name, rot0=qtn0, upright=upright, rot_type="qtn")