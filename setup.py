# setup.py
from setuptools import setup, find_packages

setup(
    name="powergpt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "tiktoken",
        "datasets",
        "tqdm",
        "pyyaml",
        "wandb",
        "tensorboard",
    ],
    entry_points={
        "console_scripts": [
            "powergpt = powergpt.cli:main",
        ],
    },
    python_requires=">=3.8",
)