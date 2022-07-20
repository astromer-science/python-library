import requests
import zipfile
import os

def download_weights(url, target):

    if not os.path.isdir(target):
        os.makedirs(target, exist_ok=True)

    r = requests.get(url)

    path_zip = '{}.zip'.format(target)
    with open(path_zip, 'wb') as f:
        f.write(r.content)

    with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        zip_ref.extractall('/'.join(path_zip.split('/')[:-1]))

    os.remove(path_zip)
