import requests
import zipfile
import pathlib
import os


def download_weights(url, target):
    """
        This method delivers the weights requested in the SingleBandEncoder class using the method 'from_pretraining()’ that specifies the available surveys; 'macho', 'atlas' and 'ztfg'.
        The UTILS module it's a set of functions that allow performing functions not considered in models and preprocessing.

        This code provides a simple and convenient way to download and extract zipped files from a URL to a specified directory using Python.

        :param url:
        :param target:

    """

    if not os.path.isdir(target):
        os.makedirs(target, exist_ok=True)

    r = requests.get(url)

    path_zip = '{}.zip'.format(target)
    with open(path_zip, 'wb') as f:
        f.write(r.content)
    
    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall(pathlib.Path(path_zip).parent.resolve())

    os.remove(path_zip)
