from setuptools import find_packages, setup

setup(
    name="microbackbone",
    version="0.1.0",
    description="MicroSign-Net backbone for microcontrollers",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "pyyaml",
        "pillow",
    ],
)
