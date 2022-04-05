from setuptools import setup, find_packages
import codecs
import os

with open("README.md", "r") as file:
    long_description = file.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Setting up
setup(
    name="deepweights",
    version='0.0.1',
    url='https://github.com/astromer-science/python-library',
    author="Cristobal Donoso",
    author_email="<cridonoso@inf.udec.cl>",
    description='ASTROMER embedding model',
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=required,
    keywords=['python', 'light curves', 'photometry', 'transformers', 'deep learning'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
# python setup.py sdist
# python setup.py bdist_wheel sdist
# twine upload dist/*
