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
import os

class CameraHandler:
    def __init__(
        self,
        cam_position: np.ndarray, 
        target_position: np.ndarray,
        world: BulletWorld,
        visualize_cam: bool = True,
    ):
        self.cam = world.add_camera(
            intrinsic=CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0),
            near=0.1,
            far=2.0
        )
        self.extrinsic = Transform.look_at(cam_position, target_position, [0,0,1])
        self.cam_pose = self.extrinsic.inverse()

        #visualize
        if visualize_cam:
            scene_maker = BulletSceneMaker(world)
            scene_maker.create_box(
                body_name="cam", 
                half_extents=[0.02]*3,
                rgba_color=[0.9,0.9,0.9,0.2],
                pose=self.extrinsic.inverse(),
                mass=1,
                ghost=True,
            )
            scene_maker.view_frame(self.extrinsic.inverse(), "cam_frame", length=0.02)

    def render(self):
        return self.cam.render(self.extrinsic)

def watch_workspace(world, target_position: np.ndarray):
    world.physics_client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=np.pi/2,
        cameraPitch=-10,
        cameraTargetPosition=target_position
    )

def main(obj_name: str, gt: np.ndarray, pred: np.ndarray, pred_type: str, filename: str, gui=True):
    """_summary_

    Args:
        obj_name (str): A name of the ycb object
        gt (np.ndarray): Ground truth quaternion
        pred (np.ndarray): Rotation prediction from the network
        pred_type (str): Representation Type ("quat"/"6d")
    """
    if os.path.exists(f"data/visualize/{obj_name}_gt_{filename}.jpg"):
        print("there is already a file that has the same filename")
    if os.path.exists(f"data/visualize/{obj_name}_pred_{filename}.jpg"):
        print("there is already a file that has the same filename")
    
    if pred_type == "quat":
        pred = Rotation.from_quat(pred)
    elif pred_type == "6d":
        pred = pred.reshape(3,3, order="F")
        pred = Rotation.from_matrix(pred)
    obj_path = "ycb/"+ obj_name +"/google_16k"

    #create environment
    world = BulletWorld(gui=gui)
    sm = BulletSceneMaker(world)
    world.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    grasp_frame = Transform(Rotation.from_euler("xyz", [0,np.pi/2, 0]), [0,0,0])
    print(grasp_frame.rotation.inv().as_quat())
    hand = world.load_urdf("hand", "data/urdfs/panda/panda_hand.urdf")
    
    #set camera
    target_point = [0, 0, 0]
    cam_position = [0.3, -0.2, 0.2]
    watch_workspace(world, target_point)
    cam = CameraHandler(cam_position, target_point, world, sm)   

    #set hand
    base_to_tcp = hand.get_link_pose(2)
    hand.set_base_pose(grasp_frame * base_to_tcp.inverse())
    
    # view ground truth 
    T_grasp_obj = Transform(Rotation.from_quat(gt), [0,0,0])
    obj = world.load_ycb(
        name="obj", 
        path=obj_path, 
        pose=grasp_frame*T_grasp_obj
    )
    
    # ground truth scene
    rgb, _ = cam.render()
    im = Image.fromarray(np.uint8(rgb))
    im.save(f"data/visualize/{obj_name}_gt_{filename}.jpg")
    
    T_upright = T_grasp_obj * Transform(pred, [0,0,0])
    obj.set_base_pose(grasp_frame*T_upright)
    rgb, _ = cam.render()
    im = Image.fromarray(np.uint8(rgb))
    im.save(f"data/visualize/{obj_name}_pred_{filename}.jpg")
    
if __name__ == "__main__":
    obj_name = "005_tomato_soup_can"
    # init rotation
    pred = np.array(
        [0.0308963917195796,-0.0136744258925318,-8.026557043194771e-05,0.9994290471076964,]
    )
    # estimated upright orientation
    gt = np.array(
        [0.5156777266951201,-0.4643074190015287,-0.4859241116089076,0.5313876744974294,]
    )
    # please specify the name of the file
    # it will be saved as "data/visualize/{obj_name}_{gt/pred}_{filename}"
    filename = "1"

    main(obj_name, gt=gt, pred=pred, pred_type="quat", filename=filename, gui=False)