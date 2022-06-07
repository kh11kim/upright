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

def main(obj_name: str, qtn: Iterable):
    obj_path = "ycb/"+ obj_name +"/google_16k"

    #create environment
    world = BulletWorld(gui=True)
    sm = BulletSceneMaker(world)
    world.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    grasp_frame = Transform(Rotation.from_euler("xyz", [0,np.pi/2, 0]), [0,0,0])
    hand = world.load_urdf("hand", "data/urdfs/panda/panda_hand.urdf")
    
    base_to_tcp = hand.get_link_pose(2)
    hand.set_base_pose(grasp_frame * base_to_tcp.inverse())
    
    #set scene
    target_point = [0, 0, 0]
    watch_workspace(world, target_point)
    
    
    sm.view_frame(grasp_frame, "grasp")

    obj = world.load_ycb(
        name="obj", 
        path=obj_path, 
        pose=grasp_frame*Transform(Rotation.from_quat(qtn), [0,0,0])
    )
    input()
    
if __name__ == "__main__":
    obj_name = "004_sugar_box"
    qtn = np.array([-0.7755,0.0275,0.0609,-0.6277])
    main(obj_name, qtn=qtn)