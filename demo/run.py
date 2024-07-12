# standard library
from typing import *
from pathlib import Path
# third party
import numpy as np
import cv2
import argparse
from tqdm import tqdm
# mp_hamer
import mp_hamer

# create pipeline
pipeline = mp_hamer.Pipeline()


# for single image
def predict_image(args):
    # read image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    H, W = image.shape[:2]
    if args.focal is None:
        args.focal = max(H, W)

    # inference
    results = pipeline(rgb_image=image, focal_length=args.focal)
    
    # render
    renderer = mp_hamer.Renderer(W, H, args.focal)
    for i, result in enumerate(results):
        # create hand mesh
        mesh = pipeline.create_trimesh(
            verts=result['verts'] * np.array([1, -1, -1]), # opencv to openGL
            is_right=result['is_right']
        )
        # add to scene
        renderer.add(
            name=f'hand_{i}',
            mesh=renderer.from_trimesh(mesh, mp_hamer.HAND_MATERIAL)
        )
    
    # offscreen render
    image = renderer.render_image(image_rgb=image)
    cv2.imwrite(args.out, image[:,:,::-1])


# for video
def predict_video(args):
    # init video reader
    cap = cv2.VideoCapture(args.video)
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FPS = int(cap.get(cv2.CAP_PROP_FPS))
    if args.focal is None:
        args.focal = max(H, W)
    
    # init renderer
    renderer = mp_hamer.Renderer(W, H, args.focal)

    # create video writer
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    hand_node = None

    ### iterate inference
    for _ in tqdm(range(TOTAL)):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # infer
        results = pipeline(rgb_image=frame, focal_length=args.focal)
        for result in results:
            verts, is_right = result['verts'], result['is_right']
            # remove previous
            renderer.remove(hand_node)
            # add new
            mesh = pipeline.create_trimesh(verts * np.array([1, -1, -1]), is_right)
            hand_node = renderer.add(
                name=f'hand_{int(is_right)}',
                mesh=renderer.from_trimesh(mesh, mp_hamer.HAND_MATERIAL)
            )

        # offscreen render
        image = renderer.render_image(image_rgb=frame)[:, :, ::-1] # RGB to BGR
        out.write(image)
    
    # release resources
    cap.release()
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=None, help='Path to input image')
    parser.add_argument('--video', type=str, default=None, help='Path to input video')
    parser.add_argument('--out', type=str, required=True, help='Path to output image/video')
    parser.add_argument('--focal', type=float, default=None, help='Focal length of camera')
    args = parser.parse_args()

    if args.image and args.video:
        raise ValueError("Cannot specify both image and video")
    
    if args.image:
        predict_image(args)
    else:
        predict_video(args)
