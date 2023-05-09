import json
import math
import os
import random

import cv2
import numpy as np
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer

from dataset.utils import pre_caption


class MiX:
    def __init__(
        self,
        img_aug_function,
        txt_aug_function,
        normal_image_root,
        normal_transform,
        image_info,
        obj_bg_threshold=0.01,
        bg_center_threshold=0.7,
    ):
        self.img_aug = img_aug_function
        self.txt_aug = txt_aug_function
        self.normal_transform = normal_transform
        self.normal_image_root = normal_image_root

        self.image_info_dict, self.obj_bg_dict = self.romixgen_preset(
            self.use_sub_dict_key(
                json.load(
                    open(image_info, "r"),
                ),
                [
                    "file_name",
                    "width",
                    "height",
                    "max_obj_super_cat",
                    "max_obj_cat",
                    "max_obj_area",
                    "max_obj_midpoint",
                    "max_obj_bbox",
                    "max_obj_area_portion",
                    "captions",
                ],
            ),
            obj_bg_threshold,
            bg_center_threshold,
        )

        # after initializing img_infodict and obj_bg_dict, pass them to romixgen_img, and romixgen_txt

        self.img_aug.__get_dict__(
            self.use_sub_dict_key(
                self.image_info_dict, ["file_name", "max_obj_midpoint", "max_obj_bbox"]
            )
        )
        self.txt_aug.__get_dict__(
            self.use_sub_dict_key(
                self.image_info_dict,
                ["file_name", "max_obj_super_cat", "max_obj_cat", "captions"],
            )
        )

    # json 파일에서 필요한 key만 뽑아서 사용
    def use_sub_dict_key(self, dic: dict, key_to_use: list):
        return_dic = {}
        for item in dic:
            for sub_item in dic[item]:
                if sub_item in key_to_use:
                    return_dic[item] = return_dic.get(item, {})
                    return_dic[item][sub_item] = dic[item][sub_item]

        return return_dic

    # bg_center_threshold 이용하는 함수
    def center_check(self, midpoint: list, width: int, height: int, thrs: float):
        width_area = [0 + width * ((1 - thrs) / 2), width - width * ((1 - thrs) / 2)]
        height_area = [
            0 + height * ((1 - thrs) / 2),
            height - height * ((1 - thrs) / 2),
        ]
        try:
            if midpoint[0] > width_area[0] and midpoint[0] < width_area[1]:
                if midpoint[1] > height_area[0] and midpoint[1] < height_area[1]:
                    return True
                else:
                    return False
            else:
                return False
        except:
            return False

    # romixgen 적용한 dictionary를 만들어주는 함수
    def romixgen_preset(
        self, img_info_dict, obj_bg_threshold=0.01, bg_center_threshold=0.7
    ):
        seg_or_bbox = "bbox"
        for key in img_info_dict.keys():
            try:
                max_obj_area_portion = img_info_dict[key]["max_obj_area_portion"]
                max_obj_midpoint = img_info_dict[key]["max_obj_midpoint"]
                img_width, img_height = int(img_info_dict[key]["width"]), int(
                    img_info_dict[key]["height"]
                )

                if max_obj_area_portion:
                    img_info_dict[key].pop(
                        "max_obj_segment_points", None
                    ) if seg_or_bbox == "bbox" else img_info_dict[key].pop(
                        "max_obj_bbox", None
                    )

                    if (
                        max_obj_area_portion > obj_bg_threshold
                    ):  # 물체가 이미지의 일정 비율 이상 차지하는 경우
                        img_info_dict[key]["obj_bg"] = "obj"
                    else:
                        if self.center_check(
                            max_obj_midpoint, img_width, img_height, bg_center_threshold
                        ):  # 빈 곳의 중심이 이미지의 중심에 가까운 경우
                            img_info_dict[key]["obj_bg"] = "bg"
                        else:  # 빈 곳의 중심이 이미지의 중심에 가깝지 않은 경우 (외곽에 위치한 경우)
                            img_info_dict[key]["obj_bg"] = "unusable_bg"

                else:  # max obj 비어있는 경우 (그냥 real bg로 저장)
                    img_info_dict[key]["obj_bg"] = "unusable_bg"

            except Exception as e:
                print(f"Error processing image {img_info_dict[key]['file_name']}: {e}")
        obj_bg_dict = {}

        # obj,bg key를 가진 dictionary 생성
        for key in img_info_dict.keys():
            if img_info_dict[key]["obj_bg"] not in obj_bg_dict:
                obj_bg_dict[img_info_dict[key]["obj_bg"]] = [key]
            else:
                obj_bg_dict[img_info_dict[key]["obj_bg"]].append(key)

        return img_info_dict, obj_bg_dict

    def normal_load(self, ann):
        image_path = os.path.join(self.normal_image_root, ann["image"])
        image = Image.open(image_path).convert("RGB")
        image = self.normal_transform(image)
        caption = pre_caption(ann["caption"], 50)  # 50 = max_words
        return image, caption

    def mix(self, obj_id, bg_id):
        img = self.img_aug(obj_id, bg_id)
        txt = self.txt_aug(obj_id, bg_id)
        return img, txt

    def select_id(self, image_id, obj_bg):
        if obj_bg == "obj":
            obj_id = image_id
            bg_id = random.choice(self.obj_bg_dict["bg"])
        elif obj_bg == "bg":
            bg_id = image_id
            obj_id = random.choice(self.obj_bg_dict["obj"])

        return obj_id, bg_id

    def __call__(self, ann):  # ann  = image_caption[index]
        image_id = ann["image_id"].split("_")[-1]
        self.image_id = image_id

        # Typeerror : #obj 혹은 bg에 bbox annotation 정보가 없는 경우
        # ValueError : #objet image의 크기가 너무 커서 resize해도 bg로 들어가지 않는 경우
        # UnboundLocalError: #obj 또는 bg 에 unusable_bg가 걸리는 경우

        try:
            obj_id, bg_id = self.select_id(
                image_id, self.image_info_dict[image_id]["obj_bg"]
            )
            img, caption = self.mix(obj_id, bg_id)

        except (TypeError, ValueError, UnboundLocalError):
            try:
                obj_id, bg_id = self.select_id(
                    image_id, self.image_info_dict[image_id]["obj_bg"]
                )
                img, caption = self.mix(obj_id, bg_id)

            except (TypeError, ValueError, UnboundLocalError):
                try:
                    obj_id, bg_id = self.select_id(
                        image_id, self.image_info_dict[image_id]["obj_bg"]
                    )
                    img, caption = self.mix(obj_id, bg_id)

                except (TypeError, ValueError, UnboundLocalError):
                    img, caption = self.normal_load(
                        ann
                    )  # self.ann 에 들어있는 ann은 caption이 한개씩 있어서 random choice 하지 않아도 됨

                    # caption = np.random.choice(caption,1)[0]
        return img, caption


class RoMixGen_Img:
    def __init__(self, image_root, transform_after_mix, midset_bool, resize_ratio=1):
        # Image
        self.image_root = image_root  # preprocessed image for augmentation root
        self.transform_after_mix = (
            transform_after_mix  # transforms functions after augmentation
        )
        self.midset_bool = midset_bool  # midset_bool = True :
        self.resize_ratio = resize_ratio

    # dataset json
    def __get_dict__(self, img_info):
        self.img_info_dict = img_info  # how large obj image resized

    def bbox_point(self, bboxes):
        y_up = int(bboxes[1])
        y_down = y_up + int(bboxes[3])
        x_left = int(bboxes[0])
        x_right = x_left + int(bboxes[2])
        return x_left, x_right, y_up, y_down

    def __cut_obj__(self, bboxes, obj_img):
        x_left, x_right, y_up, y_down = self.bbox_point(bboxes)
        obj_img = np.array(obj_img)[y_up:y_down, x_left:x_right, :]
        return obj_img

    def get_xy_point(self, bg_img, obj_img, bg_inform):
        (bg_y, bg_x, _) = np.array(bg_img).shape
        (obj_y, obj_x, _) = np.array(obj_img).shape
        [bg_midpoint_x, bg_midpoint_y] = bg_inform["max_obj_midpoint"]

        # 검정박스 우측 아래 -> 우측 아래 코너에 맞춰야 함
        if (bg_midpoint_y > bg_y / 2) & (bg_midpoint_x > bg_x / 2):
            # 오른쪽 아래
            y = bg_inform["max_obj_bbox"][1] + bg_inform["max_obj_bbox"][3]
            x = bg_inform["max_obj_bbox"][0] + bg_inform["max_obj_bbox"][2]
            x, y = x - obj_x, y - obj_y
        # 검정박스 좌측 아래 -> 좌측 아래 코너에 맞춰야 함
        elif (bg_midpoint_y > bg_y / 2) & (bg_midpoint_x < bg_x / 2):
            # 좌측 아래
            y = bg_inform["max_obj_bbox"][1] + bg_inform["max_obj_bbox"][3]
            x = bg_inform["max_obj_bbox"][0]
            y = y - obj_y

        # 검정박스 우측 위 -> 우측 위 코너에 맞춰야 함
        elif (bg_midpoint_y < bg_y / 2) & (bg_midpoint_x > bg_x / 2):
            # 오른쪽 위
            y = bg_inform["max_obj_bbox"][1]
            x = bg_inform["max_obj_bbox"][0] + bg_inform["max_obj_bbox"][2]
            x = x - obj_x

        # 검정박스 좌측 위 -> 좌측 위 코너에 맞춰야 함
        else:
            # 좌측 위
            y = bg_inform["max_obj_bbox"][1]
            x = bg_inform["max_obj_bbox"][0]

        return x, y

    def __call__(self, obj_id, bg_id):
        self.obj_inform = self.img_info_dict[obj_id]  # 이미지 전처리 정보 중 해당 obj의 정보를 가져 옴
        self.bg_inform = self.img_info_dict[bg_id]  # 이미지 전처리 정보 중 해당 bg의 정보를 가져 옴

        # image open
        obj_img = Image.open(
            os.path.join(self.image_root, "obj", self.obj_inform["file_name"])
        ).convert("RGB")
        bg_img = Image.open(
            os.path.join(self.image_root, "bg", self.bg_inform["file_name"])
        ).convert("RGB")

        obj_img = Image.fromarray(
            self.__cut_obj__(self.obj_inform["max_obj_bbox"], obj_img)
        )  # obj img cuttting

        # get paste point and paste
        if self.midset_bool:  # midset bool 이 true면 bg 이미지의 빈칸에 맞추어서 paste 함
            x, y = self.get_xy_point(bg_img, obj_img, self.bg_inform)
        else:  # 아니라면
            x, y = self.obj_inform["bbox"][0], self.obj_inform["bbox"][1]

        bg_img.paste(obj_img, (int(x), int(y)))

        # transforms after mix
        img = self.transform_after_mix(bg_img)

        return img


class RoMixGen_Txt:
    def __init__(
        self,
        first_model_name="Helsinki-NLP/opus-mt-en-fr",
        second_model_name="Helsinki-NLP/opus-mt-fr-en",
    ):
        self.first_model_tokenizer = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model = MarianMTModel.from_pretrained(first_model_name)
        self.second_model_tokenizer = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model = MarianMTModel.from_pretrained(second_model_name)

        # self.image_caption          = image_caption

    def __get_dict__(self, img_info):
        self.img_info_dict = img_info

    def back_translate(self, text):
        first_formed_text = f">>fr<< {text}"
        first_translated = self.first_model.generate(
            **self.first_model_tokenizer(
                first_formed_text, return_tensors="pt", padding=True
            )
        )
        first_translated_text = self.first_model_tokenizer.decode(
            first_translated[0], skip_special_tokens=True
        )
        second_formed_text = f">>en<< {first_translated_text}"
        second_translated = self.second_model.generate(
            **self.second_model_tokenizer(
                second_formed_text, return_tensors="pt", padding=True
            )
        )
        second_translated_text = self.second_model_tokenizer.decode(
            second_translated[0], skip_special_tokens=True
        )
        return second_translated_text

    def replace_word(self, captions, bg_cats, obj_cats):
        caption = np.random.choice(captions, 1)[0]
        try:
            (bg_cat_id, bg_cat) = list(
                filter(lambda x: x[1] in caption.lower(), enumerate(bg_cats))
            )[0]
            caption = caption.lower().replace(bg_cat, obj_cats[bg_cat_id]).capitalize()
        except IndexError:
            caption = random.choice(obj_cats) + " " + caption
        return caption

    def __call__(self, obj_id, bg_id):
        obj_cat = (
            self.img_info_dict[obj_id]["max_obj_cat"]
            + self.img_info_dict[obj_id]["max_obj_super_cat"]
        )
        bg_cat = (
            self.img_info_dict[bg_id]["max_obj_cat"]
            + self.img_info_dict[bg_id]["max_obj_super_cat"]
        )
        bg_caption = self.img_info_dict[bg_id]["captions"]
        new_caption = self.replace_word(bg_caption, bg_cat, obj_cat)
        return new_caption
        backtranslated_text = self.back_translate(new_caption)
        return backtranslated_text


class BackTranslation:
    def __init__(
        self,
        device,
        first_model_name="Helsinki-NLP/opus-mt-en-fr",
        second_model_name="Helsinki-NLP/opus-mt-fr-en",
    ):
        self.first_model_tokenizer = MarianTokenizer.from_pretrained(first_model_name)
        self.first_model = MarianMTModel.from_pretrained(first_model_name).to(device)
        self.second_model_tokenizer = MarianTokenizer.from_pretrained(second_model_name)
        self.second_model = MarianMTModel.from_pretrained(second_model_name).to(device)
        self.device = device

    def __call__(self, text):
        first_formed_text = [f">>fr<< {text}" for text in list(text)]
        first_token = self.first_model_tokenizer(
            first_formed_text, return_tensors="pt", padding=True
        ).to(self.device)
        first_translated = self.first_model.generate(**first_token)
        first_translated_text = self.first_model_tokenizer.batch_decode(
            first_translated, skip_special_tokens=True
        )
        second_formed_text = [f">>en<< {text}" for text in list(first_translated_text)]
        second_token = self.second_model_tokenizer(
            second_formed_text, return_tensors="pt", padding=True
        ).to(self.device)
        second_translated = self.second_model.generate(**second_token)
        second_translated_text = self.second_model_tokenizer.batch_decode(
            second_translated, skip_special_tokens=True
        )
        return second_translated_text
