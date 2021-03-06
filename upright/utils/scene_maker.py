import numpy as np
from typing import Any, Dict, Optional
from upright.utils.world import BulletWorld, Body
from upright.utils.transform import Rotation, Transform
import pybullet as p

"""BulletSceneMaker : Helper class to make objects.

"""
class BulletSceneMaker:
    def __init__(self, world: BulletWorld):
        self.world = world
        self.physics_client = self.world.physics_client

    def create_box(
        self,
        body_name: str,
        half_extents: np.ndarray,
        mass: float,
        pose: Transform,
        rgba_color: Optional[np.ndarray] = np.ones(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        texture: Optional[str] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=pose.translation,
            orientation=pose.rotation.as_quat(),
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        # if texture is not None:
        #     texture_path = os.path.join(get_data_path(), texture)
        #     texture_uid = self.physics_client.loadTexture(texture_path)
        #     self.physics_client.changeVisualShape(self.bullet._bodies_idx[body_name], -1, textureUniqueId=texture_uid)

    def create_cylinder(
        self,
        body_name: str,
        radius: float,
        height: float,
        mass: float,
        position: np.ndarray,
        orientation: None,
        rgba_color: Optional[np.ndarray] = np.zeros(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            orientation=orientation,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
        self,
        body_name: str,
        radius: float,
        mass: float,
        position: np.ndarray,
        rgba_color: Optional[np.ndarray] = np.zeros(4),
        specular_color: np.ndarray = np.zeros(3),
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name: str,
        geom_type: int,
        mass: float = 0.0,
        position: np.ndarray = np.zeros(3),
        orientation = None,
        ghost: bool = False,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
        visual_kwargs: Dict[str, Any] = {},
        collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        uid = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
            baseOrientation=orientation,
        )
        body = Body(self.physics_client, uid)
        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)
        self.world.register_body(body_name, body)

    def create_plane(self, z_offset: float = 0) -> None:
        """Create a plane. (Actually, it is a thin box.)

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            mass=0.0,
            pose=Transform(translation=[0.0, 0.0, z_offset - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.85, 0.85, 0.85, 1.0]),
        )

    def create_table(
        self,
        length: float,
        width: float,
        height: float,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
        lateral_friction: Optional[float] = None,
        spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a fixed table. Top is z=0, centered in y.

        Args:
            length (float): The length of the table (x direction).
            width (float): The width of the table (y direction)
            height (float): The height of the table.
            x_offset (float, optional): The offet in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            mass=0.0,
            position=np.array([x_offset, y_offset, -height / 2 + z_offset]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 0.5]),
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
        )
    
    def make_sphere_obstacle(self, name, position, rgb_color=[0.,0.,1.]):
        if not name in self.world.bodies:
            self.create_sphere(
                body_name=name,
                radius=0.02,
                mass=0.0,
                position=position,
                rgba_color=[*rgb_color,0.3],
                ghost=False
            )
        else:
            body = self.world.bodies[name]
            pose = Transform(Rotation.Identity(), position)
            body.set_base_pose(name, pose)
    
    def view_point(self, name, position, size=0.02, rgb_color=[1.,0.,0.]):
        if not name in self.world.bodies:
            self.create_sphere(
                body_name=name,
                radius=size,
                mass=0.0,
                ghost=True,
                position=position,
                rgba_color=np.array([*rgb_color, 0.3]),
            )
        else:
            body = self.world.bodies[name]
            pose = Transform(Rotation.identity(), position)
            body.set_base_pose(pose)
    
    def view_frame(
        self, 
        pose: Transform, 
        name: Optional[str] = None,
        length: float = 0.05
    ): #pos, orn
        if name is None:
            name = "frame"
        if not name in self.world.bodies:
            self.world.bodies[name] = self._make_axes(length=length)

        x_orn = p.getQuaternionFromEuler([0., np.pi/2, 0])
        y_orn = p.getQuaternionFromEuler([-np.pi/2, 0, 0])
        z_orn = [0., 0., 0., 1.]
        axis_orn = [x_orn, y_orn, z_orn]
        pos, orn = pose.translation, pose.rotation.as_quat()
        for i, idx in enumerate(self.world.bodies[name]):
            #orn_ = orn * axis_orn[i]
            _, orn_ = p.multiplyTransforms([0,0,0], orn, [0,0,0], axis_orn[i])
            #(orientation@axis_orn[i]).to_qtn()
            self.physics_client.resetBasePositionAndOrientation(
                bodyUniqueId=idx, posObj=pos, ornObj=orn_
            )

    def _make_axes(
        self,
        length=0.05
    ):
        radius = length/12
        visualFramePosition = [0,0,length/2]
        r, g, b = np.eye(3)
        orns = [
            [0, 0.7071, 0, 0.7071],
            [-0.7071, 0, 0, 0.7071],
            [0,0,0,1]
        ]
        a = 0.9
        shape_ids = []
        for color in [r, g, b]:
            shape_ids.append(
                self.physics_client.createVisualShape(
                    shapeType=self.physics_client.GEOM_CYLINDER,
                    radius=radius,
                    length=length,
                    visualFramePosition=visualFramePosition,
                    rgbaColor=[*color, a],
                    specularColor=[0., 0., 0.]
                )
            )
        axes_id = []
        for orn, shape in zip(orns, shape_ids):
            axes_id.append(
                self.physics_client.createMultiBody(
                    baseVisualShapeIndex=shape,
                    baseCollisionShapeIndex=-1,
                    baseMass=0.,
                    basePosition=[0,0,0],
                    baseOrientation=orn
                )
            )
        return axes_id