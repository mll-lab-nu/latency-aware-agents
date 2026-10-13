from pathlib import Path
from setuptools import find_packages, setup


long_description = (Path(__file__).parent / "README.md").read_text()

core_requirements = [
    "gymnasium==0.29.1",
    "rich>=13.7.1",
    "tqdm==4.67.3",
    "mujoco==3.3.0",
    "dm_control==1.0.28",
    "imageio>=2.34.2",
    "torch==2.11.0",
    "opencv-python-headless==4.10.0.84",
    "natsort==8.4.0",
]

setup(
    name="humanoid_bench",
    version="0.2",
    author="RLL at UC Berkeley",
    url="https://github.com/carlosferrazza/humanoid-bench",
    description="Humanoid Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=core_requirements,
    extras_require={
        "mjx": ["mujoco-mjx==3.3.0", "gymnax==0.0.8", "brax==0.9.4"],
    },
)
