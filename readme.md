# mp_hamer

This repo is for realtime hand pose estimation and reconstruction, based on [MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) and [HaMeR](https://github.com/geopavlakos/hamer).

### installation
```bash
### clone the repo
cd mp_hamer

# create conda env (other python versions may also work)
conda create -n mp_hamer python=3.9
conda activate mp_hamer

# install torch (for cuda 11.7, other verions may also work)
pip install torch==2.0.0 torchvision==0.15.1

# install package
pip install -e .
```

*If you want to install specific version of `pytorch` and `cuda`, check [this link](https://pytorch.org/get-started/previous-versions/).*

### usage
###### 1. pretrained HaMeR model
Download [HaMeR pretrained model](https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz) and unpack it.

###### 2. run
```bash
```

