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
    install_requires=["numpy"],  # Add dependencies like ["numpy"]
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
