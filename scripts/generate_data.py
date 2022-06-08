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
        cameraYaw=20,
        cameraPitch=-10,
        cameraTargetPosition=target_position
    )

def get_predefined_joint_config(robot: Panda):
    target_point = [0.5, 0, 0.5]
    ee_pose1 = Transform(rotation=Rotation.from_euler("zxy", [0, np.pi/3*2, np.pi/2]), translation=target_point)
    ee_pose2 = Transform(rotation=Rotation.from_euler("zxy", [0, np.pi/3, np.pi/2]), translation=target_point)
    ee_pose3 = Transform(rotation=Rotation.from_euler("zxy", [0, 0, np.pi/2]), translation=target_point)
    joint_configs = []
    for pose in [ee_pose3, ee_pose2, ee_pose1]:
        angles = robot.inverse_kinematics(pose=pose)
        if angles is not None:
            joint_configs.append(angles)
    return joint_configs

def get_grasp_config(obj_height: float, robot: Panda, obj: Body):
    #np.random.seed(2)
    z_offset = np.random.uniform(low=0, high=obj_height) - obj_height/2
    azi, alt = np.random.uniform(0, np.pi*2), np.random.uniform(0, np.pi/2)
    R_rot = Rotation.from_euler("xyz", [-alt,0,azi])
    R_rot_azi_only = Rotation.from_euler("xyz", [0,0,azi])
    #grasp_frame
    T_obj_grasp0 = Transform(rotation=Rotation.from_euler("xyz", [0, np.pi/2, 0]), translation=[0,0,z_offset])
    T_obj_grasp = Transform(R_rot * T_obj_grasp0.rotation, translation=T_obj_grasp0.translation)
    T_obj_grasp_azi_only = Transform(R_rot_azi_only * T_obj_grasp0.rotation, translation=T_obj_grasp0.translation)
    T_obj = Transform(translation=[0.5, 0, 0.5])
    yaw = np.random.uniform(0, np.pi*2)
    
    #robot.get_ee_pose() * T_obj_grasp.inverse()
    T_grasp_obj = T_obj_grasp.inverse()
    T_grasp_obj_assigned = Transform(Rotation.from_euler("zyx",[0,0,yaw])) * T_grasp_obj
    T_grasp_obj_assigned_alt_only = Transform(Rotation.from_euler("zyx",[0,0,yaw])) * T_obj_grasp_azi_only.inverse()
    T_obj = robot.get_ee_pose() * T_grasp_obj_assigned
    T_upright_orn = robot.get_ee_pose() * T_grasp_obj_assigned_alt_only
    obj.set_base_pose(T_obj)
    return T_grasp_obj_assigned, T_grasp_obj_assigned_alt_only

def write_setup(root: str, T_grasp_obj: Transform, T_grasp_upright:Transform, scene_id: str):
    csv_path = Path(root) / "grasp_pose.csv"
    if not csv_path.exists():
        #create a file
        col = ["scene_id", "ornx", "orny", "ornz", "ornw", "ornx0", "orny0", "ornz0", "ornw0"]
        with csv_path.open("w") as f:
            f.write(",".join(col))
            f.write("\n")

    # upright orientation label
    ornx, orny, ornz, ornw = (T_grasp_obj.inverse() * T_grasp_upright).rotation.as_quat()
    # given orientation (used for debugging)
    orn_x0, orn_y0, orn_z0, orn_w0 = T_grasp_obj.rotation.as_quat()
    #posx, posy, posz = T_grasp_upright.translation
    #ornx, orny, ornz, ornw = T_grasp_upright.rotation.as_quat()
    #write
    col = [scene_id, ornx, orny, ornz, ornw, orn_x0, orn_y0, orn_z0, orn_w0]
    with csv_path.open("a") as f:
        row = ",".join([str(arg) for arg in col])
        f.write(row)
        f.write("\n")

def main(obj_name: str, num_pose=5):
    obj_path = "ycb/"+ obj_name +"/google_16k"

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
    config_list = get_predefined_joint_config(robot)
    
    robot.set_arm_angles(config_list[0])
    
    obj = world.load_ycb(
        name="obj", 
        path=obj_path, 
        pose=Transform(Rotation.identity(), [0.5, 0, 0.5])
    )
    _, extent = obj.get_AABB(output_center_extent=True)
    for i in range(num_pose):
        T_grasp_obj, T_grasp_upright = get_grasp_config(extent[-1], robot, obj)
        # print(f"grasp pos : {T_grasp_obj.translation}")
        # print(f"grasp rot : {T_grasp_obj.rotation.as_quat()}")
        
        scene_id = uuid.uuid4().hex
        write_setup("data/images", T_grasp_obj, T_grasp_upright, scene_id)
        for i, q in enumerate(config_list):
            robot.set_arm_angles(q)
            ee_pose = robot.get_ee_pose()
            obj.set_base_pose(ee_pose*T_grasp_obj)
            rgb, _ = cam.render()
            im = Image.fromarray(np.uint8(rgb))
            im.save(f"data/images/{scene_id}_{obj_name}_{i}.jpg")

if __name__ == "__main__":
    obj_name = "004_sugar_box"
    main(obj_name, num_pose=5)