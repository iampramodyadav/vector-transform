# setup.py
from setuptools import setup, find_packages

setup(
    name="vector-transform",
    version="0.1.0",
    author="Pramod Yadav",
    author_email="pkyadav01234@gmail.com",
    description="A package for vector transformations and decorators",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/iampramodyadav/vector-transform",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "scipy>=1.7.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
