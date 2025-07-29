"""Setup script for gym-soarm-aloha package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gym-soarm",
    version="0.1.0",
    author="Masato Kawamura (masato-ka)",
    author_email="jp6uzv@gmail.com",
    description="A gymnasium environment for SO-ARM101 single-arm manipulation based on gym-aloha",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/gym-soarm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "mujoco>=2.3.7",
        "gymnasium>=0.29.1",
        "dm-control>=1.0.14",
        "imageio[ffmpeg]>=2.34.0",
        "opencv-python>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pre-commit>=3.7.0",
            "debugpy>=1.8.1",
        ],
        "test": [
            "pytest>=8.1.0",
            "pytest-cov>=5.0.0",
        ],
    },
    package_data={
        "gym_soarm_aloha": [
            "assets/*.xml",
            "assets/*.stl",
            "assets/assets/*.stl",
            "assets/assets/*.part",
        ],
    },
    include_package_data=True,
    keywords=["robotics", "reinforcement-learning", "so-arm", "single-arm", "gymnasium", "mujoco"],
    project_urls={
        "Bug Reports": "https://github.com/your-org/gym-soarm/issues",
        "Source": "https://github.com/your-org/gym-soarm",
    },
)