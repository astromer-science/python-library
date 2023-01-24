# setup.py
from keyring import get_keyring
get_keyring()
from setuptools import setup

setup()

# Using bumpver we can automatically
# version our library
# to install: python -m pip install bumpver
# bumpver init
# bumpver update --minor

# To upload our pkg we need to install:
# python -m pip install build twine

# to create wheel and tar.gz: python -m build
# to upload: twine upload -r testpypi dist/*
# to upload official: twine upload -r pypi dist/*
