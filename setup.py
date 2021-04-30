import os
from setuptools import setup, find_packages
from Cython.Build import cythonize
from Cython.Distutils import build_ext


EXCLUDE_FILES = [
    'test/*.py'
]

def get_ext_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)
    return paths



with open("README.md", "r") as fh:
    long_description = fh.read()

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""
    def run(self):

        # Import numpy here, only when headers are needed
        import numpy

        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())

        # Call original build_ext command
        build_ext.run(self)

setup(
    name="KLines",
    version="0.0.4",
    cmdclass={'build_ext':CustomBuildExtCommand},
    ext_modules=cythonize(
        get_ext_paths('klines', EXCLUDE_FILES),
        build_dir="build",
        compiler_directives={'language_level': 3}
    ),
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
    zip_safe=False
)
