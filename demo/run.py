from PIL import Image
import numpy as np
from mp_hamer import Estimator
import pyrender
import cv2
import trimesh

pipe = Estimator()

# from mp_hamer.hamer.utils.renderer import cam_crop_to_full

focal = 3000

# image = np.array(Image.open('demo/test.jpg'))
image = cv2.imread('demo/test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image.shape)

for _ in range(20):
    results = pipe(image, focal_length=focal)

if len(results) > 0:
    res = results[0]

    mesh = pipe.make_mesh(res['verts'], is_right=res['is_right'])
    mesh.export('demo/hand.obj')

    renderer = pyrender.OffscreenRenderer(
        viewport_width=image.shape[1],
        viewport_height=image.shape[0],
        point_size=1.0
    )
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(0.65098039,  0.74117647,  0.85882353)
    )
    rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)
    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                            ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
    camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                        cx=camera_center[0], cy=camera_center[1], zfar=1e12)
    
    # Create camera node and add it to pyRender scene
    scene.add(camera, pose=camera_pose)
    # add light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.0)
    scene.add(light)

    color, d = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    cv2.imwrite('demo/render.jpg', cv2.cvtColor(color, cv2.COLOR_RGBA2BGR))

    color = color.astype(np.float32) / 255.0
    input_img = image.astype(np.float32)/255.0
    input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
    input_img_overlay = input_img[:,:,:3] * (1-color[:,:,3:]) + color[:,:,:3] * color[:,:,3:]

    cv2.imwrite('demo/output.jpg', 255 * input_img_overlay[:, :, ::-1])

    