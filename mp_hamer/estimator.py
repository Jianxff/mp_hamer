# standard library
from typing import *
from pathlib import Path
# third party
import numpy as np
import torch
from PIL import Image
import cv2
import time
# hamer
from .hamer.models import load_hamer, DEFAULT_CHECKPOINT
from .hamer.utils.renderer import Renderer, cam_crop_to_full
from .hamer.datasets.utils import generate_image_patch_cv2
# mediapipe hands
from .detection.mp_hands import HandTracker

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

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

        # hamer config
        self.img_size = model_cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(model_cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(model_cfg.MODEL.IMAGE_STD)
        self.device = device

        # hand detector
        self.detector = HandTracker(
            use_gpu=torch.cuda.is_available()
        )

        # renderer
        self.renderer = Renderer(model_cfg, faces=model.mano.faces)

    
    def patch_data(
        self, 
        image_rgb:np.ndarray, 
        detection: Tuple[np.ndarray, bool]
    ) -> Dict:
        if detection is None:
            return None

        bbox, is_right = detection

        # patch bounding box
        image_patch, _ = generate_image_patch_cv2(
            img=image_rgb, c_x = bbox[0], c_y = bbox[1], bb_width=bbox[2], bb_height=bbox[3],
            patch_width=self.img_size, patch_height=self.img_size,
            do_flip=(not is_right), scale=1.0, rot=0,
            border_mode=cv2.BORDER_CONSTANT
        )

        # convert image strucutre
        # image_patch = image_patch[:, :, ::-1] # BGR to RGB
        image_patch = np.transpose(image_patch, (2, 0, 1)) # HWC to CHW
        image_patch = image_patch.astype(np.float32)

        # apply normalization
        for c in range(3):
            image_patch[c, :, :] = (image_patch[c, :, :] - self.mean[c]) / self.std[c]
        
        # convert to tensor
        image_torch = torch.from_numpy(image_patch).unsqueeze(0).half()

        return {'img' : image_torch.to(self.device)}
    
    @torch.no_grad()
    def __call__(
        self,
        rgb_image: Union[np.ndarray, str, Path],
        timestamp_ms: Optional[int] = None
    ) -> Dict:
        if isinstance(rgb_image, (str, Path)):
            rgb_image = np.array(Image.open(rgb_image))

        # detect hands
        results = self.detector(rgb_image, timestamp_ms)
        detections = self.detector.get_bbox(
            results=results, 
            image_HW=rgb_image.shape[:2], 
            extend_scale=2
        )

        if detections is None or len(detections) == 0:
            return None

        # patch data
        data = self.patch_data(rgb_image, detections[0])

        # inference
        out = self.model(data)

        return out
