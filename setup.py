from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ASTROMER',
    version='1.0.0',
    license='MIT',
    url='https://github.com/astromer-science/python-library',
    description='ASTROMER embedding model',
    author="Cristobal Donoso-Oliva",
    author_email='cridonoso@inf.udec.cl',
    py_modules=['ASTROMER/*'],
    package_dir={'':'src'},
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    extras_require = {
        "dev": [
            "pytest>=7.0.0",
            "alerce"
        ],
    },

)

# python setup.py sdist
# python setup.py bdist_wheel sdist
# twine upload dist/*
