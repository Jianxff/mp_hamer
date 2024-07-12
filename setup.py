from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='HaMeR as a package',
    name='mp_hamer',
    packages=find_packages(),
    install_requires=[
        'gdown',
        'numpy==1.26',
        'mediapipe',
        'opencv-python',
        'pyrender',
        'pytorch-lightning',
        'scikit-image',
        'smplx==0.1.28',
        'torch',
        'torchvision',
        'yacs',
        'timm',
        'einops',
        'pillow',
        'chumpy @ git+https://github.com/mattloper/chumpy'
    ]
)
