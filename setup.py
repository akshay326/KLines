from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="KLines",
    version="0.0.3",
    description="Finding K-Means for lines",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=('./examples', './tests')),
    install_requires=[
        "numpy == 1.16.5"
    ],
    license="Apache License, Version 2.0",
    url="https://github.com/be-apt/KLines",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
