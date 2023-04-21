"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from pathlib import Path
import wget
import zipfile
import shutil
from lavis.common.utils import (
    cleanup_dir,
    download_and_extract_archive,
)


DATA_URL = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",  # md5: 0da8c0bd3d6becc4dcb32757491aca88
    "val": "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
    "test": "http://images.cocodataset.org/zips/test2014.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
}


def download_datasets(root, url):
    download_and_extract_archive(url=url, download_root=root, extract_root=storage_dir)


if __name__ == "__main__":

    download_dir = Path("./data/download")
    storage_dir = Path("./data/COCO/Images/")
    storage_dir_ = Path("./data/COCO/Annotations/")
    # make sure dir exists
    download_dir.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_dir_.mkdir(parents=True, exist_ok=True)
    
    # Image download and extraction
    if storage_dir.exists() and any(file.suffix == '.jpg' for file in storage_dir.glob('*')):
        print("Images already downloaded")
    else:
        print("Downloading images to {}".format(str(storage_dir)))
        try:
            for k, v in DATA_URL.items():
                print("Downloading {} to {}".format(v, k))
                download_datasets(download_dir, v)
        except Exception as e:
            # remove download dir if failed
            cleanup_dir(download_dir)
            print("Failed to download or extracting datasets. Aborting.")

        cleanup_dir(download_dir)
        #after download move all images in each folder to coco
        #only folder
        for folder in [item for item in os.listdir(storage_dir) if os.path.isdir(os.path.join(storage_dir, item))]:
            print("Moving {}'s images to coco".format(folder))
            for file in os.listdir(storage_dir/folder):
                os.rename(storage_dir/folder/file, storage_dir/file)
            os.rmdir(storage_dir/folder)
            print("Done moving {}'s images to coco".format(folder))

    # Annotation download and extraction
    if storage_dir_.exists() and any(file.suffix == '.json' for file in storage_dir_.glob('*')):
        print("Annotations already downloaded")
    else:
        print("Downloading annotations to {}".format(str(storage_dir_)))
        wget.download("http://images.cocodataset.org/annotations/annotations_trainval2014.zip", out=str(storage_dir_))
        print("Done downloading annotations to {}".format(str(storage_dir_)))

        #unzip annotations
        print("Unzipping annotations to {}".format(str(storage_dir_)))
        with zipfile.ZipFile(storage_dir_/"annotations_trainval2014.zip", 'r') as zip_ref:
            zip_ref.extractall(storage_dir_)
        # move file to upper directory
        os.rename(storage_dir_/"annotations/instances_train2014.json", storage_dir_/"instances_train2014.json")
        os.rename(storage_dir_/"annotations/instances_val2014.json", storage_dir_/"instances_val2014.json")
        os.rename(storage_dir_/"annotations/captions_train2014.json", storage_dir_/"captions_train2014.json")
        os.rename(storage_dir_/"annotations/captions_val2014.json", storage_dir_/"captions_val2014.json")
        os.remove(storage_dir_/"annotations_trainval2014.zip")
        #remove the json file in the storage_dir
        print("Done unzipping annotations to {}".format(str(storage_dir_)))

        # remove unusing annotations from annotations_trainval2014.zip
        print("Removing unusing annotations")
        #remove folder annotations and its content
        shutil.rmtree(storage_dir_ / "annotations")
    # download albef's pretrain json of coco
    # pass if already downloaded
    if os.path.exists(storage_dir_/"coco.json"):
        print("Pretrain json already downloaded")
    else:
        print("Downloading pretrain json to {}".format(str(storage_dir_)))
        wget.download("https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip", out=str(storage_dir_))
        #unzip
        print("Unzipping pretrain json to {}".format(str(storage_dir_)))
        with zipfile.ZipFile(storage_dir_/"json_pretrain.zip", 'r') as zip_ref:
            zip_ref.extractall(storage_dir_)
        os.remove(storage_dir_/"json_pretrain.zip")
        # extract only coco.json to storage_dir_
        print("Done unzipping pretrain json to {}".format(str(storage_dir_)))
        os.rename(storage_dir_/"json_pretrain/coco.json", storage_dir_/"coco.json")
        shutil.rmtree(storage_dir_ / "json_pretrain")
        print("Done moving pretrain json to {}".format(str(storage_dir_)))
    
    print("Done downloading all datasets")
