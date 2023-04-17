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
    storage_dir = Path("./data/COCO/Images/raw_images")
    storage_dir_ = Path("./dataCOCO/Annotations")

    if storage_dir.exists():
        print(f"Dataset already exists at {storage_dir}. Aborting.")
        exit(0)

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

    #download annotations
    # pass if already downloaded
    if os.path.exists(storage_dir_/"annotations_trainval2014.zip"):
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
    for folder in [item for item in os.listdir(storage_dir_/"annotations") if os.path.isdir(os.path.join(storage_dir_/"annotations", item))]:
        os.rename(storage_dir_/"annotations"/file, storage_dir_/file)
    os.rmdir(storage_dir_/"annotations")
    
    os.remove(storage_dir_/"annotations_trainval2014.zip")
    #remove the json file in the storage_dir
    print("Done unzipping annotations to {}".format(str(storage_dir_)))