import os
from pathlib import Path
import wget
import zipfile
import shutil

IMG_DATA_URL = {
    "train2014.zip": "http://images.cocodataset.org/zips/train2014.zip",  # md5: 0da8c0bd3d6becc4dcb32757491aca88
    "val2014.zip": "http://images.cocodataset.org/zips/val2014.zip",  # md5: a3d79f5ed8d289b7a7554ce06a5782b3
    "test2014.zip": "http://images.cocodataset.org/zips/test2014.zip",  # md5: 04127eef689ceac55e3a572c2c92f264
}
ANN_DATA_URL = {
    "annotations_trainval2014.zip" : "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "json_pretrain.zip" : "https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/json_pretrain.zip",
}

DATA_PATH = {
    "download": "./data/download/",
    "images": "./data/COCO/Images/",
    "annotations" : "./data/COCO/Annotations/",
}

def download_datasets(url, download_root, extract_root, file):
    # download dataset using wget
    # if file exist in download root, skip
    if Path(download_root+file).exists():
        pass
    else:
        wget.download(url, out=download_root+file)
    # extract dataset
def unzip_datasets(download_root, extract_root, file):
    with zipfile.ZipFile(download_root+file, 'r') as zip_ref:
        zip_ref.extractall(extract_root)


if __name__ == "__main__":

    download_dir = DATA_PATH["download"]
    storage_dir_img = DATA_PATH["images"]
    storage_dir_ann = DATA_PATH["annotations"]
    # make sure dir exists
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    Path(storage_dir_img).mkdir(parents=True, exist_ok=True)
    Path(storage_dir_img).mkdir(parents=True, exist_ok=True)
    
    # coco dataset image 
    if Path(storage_dir_img).exists() and any(file.suffix == '.jpg' for file in Path(storage_dir_img).glob('*')):
        print("Images already downloaded")
    else:
        print("Downloading images to {}".format(str(storage_dir_img)))
        try:
            for k, v in IMG_DATA_URL.items():
                #if zip file exist in download dir, skip
                if Path(download_dir+k).exists():
                    print("File {} already exist in {}".format(k, download_dir))
                    pass
                else:
                    print("Downloading {} to {}".format(v, k))
                    download_datasets(v, download_dir, storage_dir_img, k)
            try:
                for k, v in IMG_DATA_URL.items():
                    print("Unzipping images to {}".format(str(storage_dir_img)))
                    unzip_datasets(download_dir, storage_dir_img, k)
            except Exception as e:
                print("Failed to unzip files. Aborting.")
        except Exception as e:
            # remove download dir if failed remove whether if file exist in download dir
            print("Failed to download files. Aborting.")
        # unzip the file
        print("Done downloading images to {}".format(str(storage_dir_img)))

        
        #after download move all images in each folder to coco
        #only folder
        for folder in [item for item in os.listdir(storage_dir_img) if os.path.isdir(os.path.join(storage_dir_img, item))]:
            print("Moving {}'s images to coco".format(folder))
            for file in os.listdir(storage_dir_img+folder):
                os.rename(storage_dir_img+folder+"/"+file, storage_dir_img+"/"+file)
            os.rmdir(storage_dir_img+folder)
            print("Done moving {}'s images to coco".format(folder))

    # Annotation download and extraction
    if Path(storage_dir_ann).exists() and any(file.suffix == '.json' for file in Path(storage_dir_ann).glob('*')):
        print("Annotations already downloaded")
    else:
        print("Downloading annotations to {}".format(str(storage_dir_ann)))
        for k, v in ANN_DATA_URL.items():
            if Path(download_dir+k).exists():
                print("File {} already exist in {}".format(k, download_dir))
                pass
            else:
                print("Downloading {} to {}".format(v, k))
                download_datasets(v, download_dir, storage_dir_ann, k)
        print("Done downloading annotations to {}".format(str(storage_dir_ann)))
        #unzip the file
        print("Unzipping annotations to {}".format(str(storage_dir_ann)))
        for k, v in ANN_DATA_URL.items():
            unzip_datasets(download_dir, storage_dir_ann, k)

        # move file to upper directory
        os.rename(storage_dir_ann+"annotations/instances_train2014.json", storage_dir_ann+"instances_train2014.json")
        os.rename(storage_dir_ann+"annotations/instances_val2014.json", storage_dir_ann+"instances_val2014.json")
        os.rename(storage_dir_ann+"annotations/captions_train2014.json", storage_dir_ann+"captions_train2014.json")
        os.rename(storage_dir_ann+"annotations/captions_val2014.json", storage_dir_ann+"captions_val2014.json")
        os.rename(storage_dir_ann+"json_pretrain/coco.json", storage_dir_ann+"coco.json")
        #remove the json file in the storage_dir
        print("Done unzipping annotations to {}".format(str(storage_dir_ann)))

        shutil.rmtree(storage_dir_ann + "json_pretrain")
        shutil.rmtree(storage_dir_ann + "annotations")

    
    print("Done downloading all datasets")