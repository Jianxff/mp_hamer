### standard library
from typing import *
### third party
import numpy as np
import pyrender
import trimesh
# import open3d as o3d
import cv2


class Renderer:
    COLOR_BLUE = (0.65, 0.74, 0.86)
    COLOR_GREEN = (0.65, 0.86, 0.74)
    COLOR_WHITE = (1.0, 1.0, 1.0)

    nodes = {}
    
    def __init__(
        self,
        width: int,
        height: int, 
        focal: float,
        znear: Optional[float] = 0.01,
        zfar: Optional[float] = 100
    ) -> None:
        # init offscreen render pipe
        self.render_pipe = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0
        )
        # init scene
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        # init camera
        camera_pose = np.eye(4)
        camera_center = [width / 2., height / 2.]
        camera = pyrender.IntrinsicsCamera(
            fx=focal, fy=focal,
            cx=camera_center[0], cy=camera_center[1], 
            znear=znear, zfar=zfar
        )
        # add it to pyRender scene
        self.cam_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(self.cam_node)
        # self.scene.add(camera, pose=camera_pose)
        # add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
        self.scene.add(light)


    def add(
        self, 
        name: str, 
        mesh: trimesh.Trimesh, 
        color: Optional[Tuple[float]] = COLOR_WHITE,
        replace: Optional[bool] = False
    ) -> pyrender.Node:
        # replace old node
        if replace:
            self.remove(name)
        # create pyrender mesh
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=color
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        # create node
        mesh_node = pyrender.Node(mesh=mesh, name=name, matrix=np.eye(4))
        # add node to scene
        self.scene.add_node(mesh_node)
        self.nodes[name] = mesh_node

        return mesh_node
    
    def set(
        self, 
        name: str, 
        pose: Optional[np.ndarray] = None,
        visible: Optional[bool] = None
    ) -> None:
        node = self.nodes.get(name, None)
        if node is not None:
            if pose is not None:
                self.scene.set_pose(node, pose)
            if visible is not None:
                node.mesh.is_visible = visible
    
    def get(self, name: str) -> pyrender.Node:
        return self.nodes.get(name, None)
    
    def remove(self, name) -> None:
        node = self.nodes.get(name, None)
        if node is not None:
            self.scene.remove_node(node)
            del self.nodes[name]

    def render(
        self
    ) -> np.ndarray:
        return self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)

    def render_mask(
        self
    ) -> np.ndarray:
        color, _ = self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        mask = (color[..., 3:] > 0).astype(np.uint8) * 255
        mask = np.squeeze(mask)
        return mask
    
    def render_sideview(
        self,
        pose_t: List[float] = [0.3, 0, -0.3],
        pose_r: List[List[float]] = [[0, 0, 1],[0, 1, 0],[-1, 0, 0]]
    ) -> np.ndarray:
        side_pose = np.eye(4)
        side_pose[:3, 3] = np.array(pose_t)
        # look to the negative x-axis
        side_pose[:3, :3] = np.array(pose_r)
        self.scene.set_pose(self.cam_node, side_pose)
        color, _ = self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        self.scene.set_pose(self.cam_node, np.eye(4))
        color = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
        return color

    def render_image(
        self,
        image_rgb: np.ndarray
    ) -> np.ndarray:
        color, _ = self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        # convert to float type
        color = color.astype(np.float32) / 255.0
        image = image_rgb.astype(np.float32) / 255.0
        # convert to RGBA
        image = np.concatenate([
            image, np.ones_like(image[..., :1])
        ], axis=2)
        # overlay render image and original image
        image_overlay = image[:,:,:3] * (1 - color[:,:,3:]) + color[:,:,:3] * color[:,:,3:]
        return (image_overlay * 255).astype(np.uint8)
    
# class Renderer:
#     geometries = {}

#     def __init__(
#         self,
#         width: int,
#         height: int, 
#         focal: float,
#         znear: Optional[float] = 0.01,
#         zfar: Optional[float] = 100
#     ) -> None:
#         # offscreen render pipe
#         self.render_pipe = o3d.visualization.rendering.OffscreenRenderer(
#             width=width, height=height
#         )
#         # camera
#         self.render_pipe.setup_camera(
#             intrinsic_matrix=np.array([
#                 [focal, 0, width / 2],
#                 [0, focal, height / 2],
#                 [0, 0, 1]
#             ]),
#             extrinsic_matrix=np.eye(4),
#             intrinsic_width_px=width,
#             intrinsic_height_px=height
#         )
#         # self.render_pipe.scene.camera.set_projection(
#         #     intrinsics=np.ndarray([
#         #         [focal, 0, width / 2],
#         #         [0, focal, height / 2],
#         #         [0, 0, 1]
#         #     ]),
#         #     near_plane=znear,
#         #     far_plane=zfar,
#         #     image_width=width,
#         #     image_height=height
#         # )
#         # scene
#         self.scene = self.render_pipe.scene
#         self.scene.set_background([0, 0, 0, 0])
#         # light
#         self.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
#         self.scene.scene.enable_sun_light(True)
#         # self.scene.add_directional_light(
#         #     name='light', color=np.array([1.0, 1.0, 1.0]),
#         #     intensity=1.5
#         # )

#     def add_geometry(
#         self,
#         name: str,
#         geometry: o3d.geometry.TriangleMesh,
#         color=[0.65, 0.74, 0.85]
#     ) -> None:
#         # convert from trimesh
#         material = o3d.visualization.rendering.MaterialRecord()
#         material.base_color = [*color, 1.0]
#         material.shader = "defaultLit"
#         # save to geometries
#         self.geometries[name] = geometry
#         # add to scene
#         self.scene.add_geometry(name, self.geometries[name], material)
#         self.geometries[name].compute_vertex_normals()

#     def get_mesh(self, name:str) -> o3d.geometry.TriangleMesh:
#         return self.geometries[name]
    
#     def update_verts(
#         self, 
#         name:str,
#         verts: np.ndarray
#     ) -> None:
#         mesh = self.geometries[name]
#         mesh.vertices = o3d.utility.Vector3dVector(verts)
#         mesh.compute_vertex_normals()
#         # update scene
#         self.scene.update_geometry(name, mesh)

#     def render(self) -> np.ndarray:
#         image = self.render_pipe.render_to_image()
#         print(image)
#         return np.asarray(image)