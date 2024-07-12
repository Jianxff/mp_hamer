### standard library
from typing import *
from pathlib import Path
### third party
import numpy as np
import torch
from PIL import Image
import cv2
import time
import trimesh
### hamer
from .hamer.models import load_hamer, DEFAULT_CHECKPOINT
# from .hamer.utils.renderer import Renderer, cam_crop_to_full
# from .hamer.datasets.utils import generate_image_patch_cv2
### mediapipe hands
from .detection.mp_hands import HandTracker

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

MANO_FACES_NEW = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                        [120, 108, 78], [78, 108, 79]])

DEFAULT_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class PoseEstimator:
    detector = None         # hand detector
    model = None            # hamer model

    def __init__(
        self, 
        checkpoint: Path = Path(DEFAULT_CHECKPOINT),
        device: Optional[torch.device] = DEFAULT_DEVICE
    ) -> None:
        # load hamer model
        model, model_cfg = load_hamer(str(checkpoint))
        model.half()
        self.model = model.to(device)
        self.model.eval()
        self.model_cfg = model_cfg

        # hamer config
        self.img_size = model_cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(model_cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(model_cfg.MODEL.IMAGE_STD)
        self.device = device

        # hand detector
        self.detector = HandTracker(use_gpu=torch.cuda.is_available())

        # mano faces
        self.mano_faces = np.concatenate([model.mano.faces, MANO_FACES_NEW], axis=0)
        self.mano_faces_left = self.mano_faces[:, [0, 2, 1]]

    
    def patch_data(
        self, 
        image_rgb: np.ndarray, 
        detection: Tuple[np.ndarray, bool],
        focal_length: float,
    ) -> Tuple[Dict, Dict]:
        bbox, is_right = detection
        H, W = image_rgb.shape[:2]

        ### patch bounding box
        # image_patch, _ = generate_image_patch_cv2(
        #     img=image_rgb, c_x = bbox[0], c_y = bbox[1], bb_width=bbox[2], bb_height=bbox[3],
        #     patch_width=self.img_size, patch_height=self.img_size,
        #     do_flip=(not is_right), scale=1.0, rot=0,
        #     border_mode=cv2.BORDER_CONSTANT
        # )
        # scale = self.img_size / max(H, W)

        ### crop and resize
        c_x, c_y, b = bbox[0], bbox[1], int(bbox[2] / 2.)
        # calculate padding
        w_left = max(0, -(c_x - b))
        w_right = max(W, c_x + b) - W
        h_top = max(0, -(c_y - b))
        h_bottom = max(H, c_y + b) - H
        padding = max([w_left, w_right, h_top, h_bottom])
        print('padding:', padding)
        # pad original image
        image_patch_full = np.zeros((H + 2 * padding, W + 2 * padding, 3), dtype=np.uint8)
        image_patch_full[padding:padding + H, padding:padding + W] = image_rgb
        # crop to bounding box
        image_patch = image_patch_full[
            c_y + padding - b:c_y + padding + b,
            c_x + padding - b:c_x + padding + b
        ]


        ### resize to model input size
        scale = self.img_size / max(image_patch.shape[:2])
        image_patch = cv2.resize(image_patch, (0, 0), fx=scale, fy=scale)
        if not is_right:
            image_patch = cv2.flip(image_patch, 1)

        ### convert image strucutre
        image_patch = np.transpose(image_patch, (2, 0, 1)) # HWC to CHW
        image_patch = image_patch.astype(np.float32)
        # apply normalization
        for c in range(3):
            image_patch[c, :, :] = (image_patch[c, :, :] - self.mean[c]) / self.std[c]
        # convert to tensor
        image_torch = torch.from_numpy(image_patch).unsqueeze(0).half()

        # make extra data
        extra = {
            'is_right': is_right,
            'box_center': (bbox[0], bbox[1]),
            'box_size': bbox[2],
            'img_size': (H, W), # HW
            'focal_length': focal_length
        }

        ### transform focal length
        scaled_focal = focal_length * scale

        return {'img' : image_torch.to(self.device), 'focal': scaled_focal}, extra
    

    @torch.no_grad()
    def __call__(
        self,
        rgb_image: Union[np.ndarray, str, Path],
        focal_length: Optional[float] = None,
        timestamp_ms: Optional[int] = None
    ) -> List[np.ndarray]:
        if isinstance(rgb_image, (str, Path)):
            rgb_image = np.array(Image.open(rgb_image))
        
        # set focal length
        if focal_length is None:
            focal_length = max(rgb_image.shape[:2])
        
        # detect hands
        results = self.detector(rgb_image, timestamp_ms)
        detections = self.detector.get_bbox(
            results=results, 
            image_HW=rgb_image.shape[:2], 
            extend_scale=1.5
        )

        if detections is None or len(detections) == 0:
            return []

        all_out = []
        for detection in detections:
            # patch data
            data, extra_info = self.patch_data(rgb_image, detection, focal_length)
            # inference
            out = self.model(data)
            # verts
            verts = self.extract_verts(out, extra_info)
            all_out.append({
                'verts': verts,
                'bbox': detection[0],
                'is_right': detection[1]
            })

        return all_out
    

    def extract_verts(
        self,
        prediction: Dict,
        extra_info: Dict
    ) -> trimesh.Trimesh:
        is_right = extra_info['is_right']

        # get vertices and pred_cam
        verts = prediction['pred_vertices'][0].detach().cpu().numpy()
        camera = prediction['pred_cam'][0].detach().cpu().numpy()
        if not is_right:
            verts[:, 0] *= -1
            camera[1] *= -1
        
        # make camera transform
        cx, cy = extra_info['box_center']
        b = extra_info['box_size']
        h, w = extra_info['img_size'] # image size, HW
        w_2, h_2 = w / 2., h / 2.
        # focal transform
        focal = extra_info['focal_length']

        bs = b * camera[0] + 1e-9
        tz = 2 * focal / bs
        tx = (2 * (cx - w_2) / bs) + camera[1]
        ty = (2 * (cy - h_2) / bs) + camera[2]

        return verts + np.array([tx, ty, tz])
    

    def make_mesh(
        self,
        verts: np.ndarray,
        is_right: bool = True,
        color = LIGHT_BLUE
    ) -> trimesh.Trimesh:
        verts_color = np.array([(*color, 1)] * verts.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices=verts, faces=self.mano_faces.copy(), vertex_colors=verts_color)
        else:
            mesh = trimesh.Trimesh(vertices=verts, faces=self.mano_faces_left.copy(), vertex_colors=verts_color)
        
        return mesh
        

    # def render_result(
    #     self,
    #     image_rgb: np.ndarray,
    #     out
    # ) -> np.ndarray:
    #     renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
    #     regression_img = renderer(
    #         out['pred_vertices'][0].detach().cpu().numpy(),
    #         out['pred_cam_t'][0].detach().cpu().numpy(),
    #         out['focal_length'][0][0],
    #         image_rgb,
    #         mesh_base_color=LIGHT_BLUE,
    #         scene_bg_color=(1, 1, 1),
    #     )

    #     return 255 * regression_img[:, :, ::-1]
        
        


