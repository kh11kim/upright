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
from typing import Iterable, Optional
from upright.inference import *
from upright.loss import *
import pandas as pd
import glob
import time
import pickle

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

def get_joint_configs(robot: Panda, num_scene: int, sm:Optional[BulletSceneMaker]=None, rot=Rotation.identity(), ):
    target_point = [0.5, 0, 0.5]
    yaw_list = np.linspace(0, np.pi/3*2, num_scene)
    ee_pos_list = []
    for yaw in yaw_list:
        tf = Transform(rotation=Rotation.from_euler("zxy", [0, yaw, np.pi/2])*rot.inv(), translation=target_point)
        ee_pos_list.append(tf)
        if sm is not None:
            sm.view_frame(tf, "1")
    
    joint_configs = []
    for pose in ee_pos_list:
        angles = robot.inverse_kinematics(pose=pose)
        if angles is not None:
            joint_configs.append(angles)
    return joint_configs

def watch_workspace(world, target_position: np.ndarray):
    world.physics_client.resetDebugVisualizerCamera(
        cameraDistance=1.0,
        cameraYaw=np.pi/2,
        cameraPitch=-10,
        cameraTargetPosition=target_position
    )

def get_grasp_config(obj_height: float, robot: Panda, obj: Body):
    np.random.seed(2)
    z_offset = -0.02 #np.random.uniform(low=0, high=obj_height) - obj_height/2
    azi, alt = np.pi/4, np.pi/6 #np.random.uniform(0, np.pi*2), np.random.uniform(0, np.pi/4)

    R_rot = Rotation.from_euler("xyz", [-alt,0,azi])
    R_rot_azi_only = Rotation.from_euler("xyz", [0,0,azi])
    #grasp_frame
    T_obj_grasp0 = Transform(rotation=Rotation.from_euler("xyz", [0, np.pi/2, 0]), translation=[0,0,z_offset])
    T_obj_grasp = Transform(R_rot * T_obj_grasp0.rotation, translation=T_obj_grasp0.translation)
    T_obj_grasp_azi_only = Transform(R_rot_azi_only * T_obj_grasp0.rotation, translation=T_obj_grasp0.translation)
    T_obj = Transform(translation=[0.5, 0, 0.5])
    yaw = np.pi/4#np.pi/2*3#np.random.uniform(0, np.pi*2)
    
    #robot.get_ee_pose() * T_obj_grasp.inverse()
    T_grasp_obj = T_obj_grasp.inverse()
    T_grasp_obj_assigned = Transform(Rotation.from_euler("zyx",[0,0,yaw])) * T_grasp_obj
    T_grasp_obj_assigned_alt_only = Transform(Rotation.from_euler("zyx",[0,0,yaw])) * T_obj_grasp_azi_only.inverse()
    T_obj = robot.get_ee_pose() * T_grasp_obj_assigned
    T_upright_orn = robot.get_ee_pose() * T_grasp_obj_assigned_alt_only
    obj.set_base_pose(T_obj)
    return T_grasp_obj_assigned, T_grasp_obj_assigned_alt_only, T_obj

class Grasp:
    def __init__(self, obj: Body, robot: Panda, grasp: Transform):
        self.obj = obj
        self.grasp = grasp
        self.robot = robot
    
    def assign(self):
        obj_pose = self.robot.get_ee_pose()*self.grasp
        self.obj.set_base_pose(obj_pose)

def main(obj_name: str):
    obj_path = "ycb/"+ obj_name +"/google_16k"
    num_scene = 3

    #create environment
    world = BulletWorld(gui=True)
    sm = BulletSceneMaker(world)
    world.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = world.load_robot("panda", robot_class=Panda)
    sm.create_plane()

    #set scene
    target_point = [0.5, 0, 0.5]
    cam_position = [0.6, -0.4, 0.7]
    watch_workspace(world, target_point)
    cam = CameraHandler(cam_position, target_point, world, sm)   
    tf = Transform.from_file("tf.p")
    rot = tf.rotation
    
    #joint_configs = get_joint_configs(robot, 3, sm, rot=rot)


    # object init pose
    T_obj_init = Transform(Rotation.identity(), [0.5, 0, 0.5])
    obj = world.load_ycb(
        name="obj", 
        path=obj_path, 
        pose=T_obj_init
    )
    _, extent = obj.get_AABB(output_center_extent=True)
    
    q_init = get_joint_configs(robot, 3)[0]
    robot.set_arm_angles(q_init)
    T_grasp_obj, T_grasp_obj_assigned_alt_only, T_obj = get_grasp_config(extent[-1], robot, obj)
    grasp = Grasp(obj, robot, T_grasp_obj)

    rot = Rotation.identity()
    for i in range(10):
        def get_images(grasp: Grasp, rot=Rotation.identity()):
            images = []
            joint_configs = get_joint_configs(robot, 3, rot=rot)
            for q in joint_configs:
                robot.set_arm_angles(q)
                grasp.assign()
                rgb, _ = cam.render()
                im = Image.fromarray(np.uint8(rgb))
                images.append(im)
                time.sleep(1)
            return get_images_from_PIL(images), joint_configs

        imgs, configs = get_images(grasp, rot=rot)

        pred = predict(model, imgs).detach().numpy()
        pred = pred[[1, 2, 3, 0]]
        print("Prediction for the images:",pred)
        rot = Rotation.from_quat(pred)
        #tf = Transform(rot, [0,0,0])
        #tf.save("tf.p")
        robot.set_arm_angles(configs[0])
        obj.set_base_pose(robot.get_ee_pose()*T_grasp_obj)
        ee_pose = robot.get_ee_pose()
        T_upright = ee_pose * T_grasp_obj * Transform(rot, [0,0,0])
        obj.set_base_pose(T_upright)
        ee_pose_new = T_upright * T_grasp_obj.inverse()
        sm.view_frame(ee_pose_new, "ee_pose_new")
        q_new = robot.inverse_kinematics(pose=ee_pose_new)
        robot.set_arm_angles(q_new)
        print("done")
        
        #
        #obj_pose = ee_pose * T_upright
        #obj.set_base_pose(obj_pose)
        #T_grasp_obj = T_upright
    
    input()

    # for i in range(10):
    #     obj.set_base_pose(obj_pose)
    #     obj.set_base_pose(ee_pose*T_grasp_obj)
    
    # data_path = "data/eval-models/data-eval/"
    # np.random.seed(2)
    # dataset = pd.read_csv(data_path + "grasp_pose.csv")
    # rand_index = np.random.choice(range(len(dataset)))
    # scene_id = dataset.iloc[rand_index,:].scene_id
    # filenames = glob.glob(data_path+scene_id+"*")
    # images = get_images_from_paths(filenames)
    # orn_col = ["orn"+a for a in ["x", "y", "z", "w"]]
    # true = dataset[dataset["scene_id"] == scene_id][orn_col].to_numpy().flatten()
    # pred = predict(model, images).detach().numpy()
    # pred = pred[[1, 2, 3, 0]]
    
    # true = Rotation.from_quat(true)
    # pred = Rotation.from_quat(pred)
    # diff = true.inv() * pred
    # angle = np.arccos((np.trace(diff.as_matrix()) - 1)/2)
    # pred = prediction.detach().numpy()

    


    if rot_type == "qtn":
        pred = Rotation.from_quat(pred)
    if rot_type == "mat":
        pred = pred.reshape(3,3, order="F")
        pred = Rotation.from_matrix(pred)
    
    

    
    
    grasp_frame = Transform(Rotation.from_euler("xyz", [0,np.pi/2, 0]), [0,0,0])
    print(grasp_frame.rotation.inv().as_quat())
    hand = world.load_urdf("hand", "data/urdfs/panda/panda_hand.urdf")
    
    base_to_tcp = hand.get_link_pose(2)
    hand.set_base_pose(grasp_frame * base_to_tcp.inverse())
    
    #set scene
    target_point = [0, 0, 0]
    watch_workspace(world, target_point)
    
    sm.view_frame(grasp_frame, "grasp")
    T_grasp_obj = Transform(Rotation.from_quat(orn0), [0,0,0])
    obj = world.load_ycb(
        name="obj", 
        path=obj_path, 
        pose=grasp_frame*T_grasp_obj
    )
    T_upright = T_grasp_obj * Transform(pred, [0,0,0])
    obj.set_base_pose(grasp_frame*T_upright)
    #pred_upright = Rotation.from_quat(qtn) * Rotation.from_quat(rot_est).inv()
    
    input()
    
if __name__ == "__main__":
    obj_name = "019_pitcher_base"
    main(obj_name)