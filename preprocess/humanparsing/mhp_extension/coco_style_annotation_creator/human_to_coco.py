import argparse
import datetime
import json
import os
from PIL import Image
import numpy as np

import pycococreatortools


def get_arguments():
    parser = argparse.ArgumentParser(description="transform mask annotation to coco annotation")
    parser.add_argument("--dataset", type=str, default='CIHP', help="name of dataset (CIHP, MHPv2 or VIP)")
    parser.add_argument("--json_save_dir", type=str, default='../data/msrcnn_finetune_annotations',
                        help="path to save coco-style annotation json file")
    parser.add_argument("--use_val", type=bool, default=False,
                        help="use train+val set for finetuning or not")
    parser.add_argument("--train_img_dir", type=str, default='../data/instance-level_human_parsing/Training/Images',
                        help="train image path")
    parser.add_argument("--train_anno_dir", type=str,
                        default='../data/instance-level_human_parsing/Training/Human_ids',
                        help="train human mask path")
    parser.add_argument("--val_img_dir", type=str, default='../data/instance-level_human_parsing/Validation/Images',
                        help="val image path")
    parser.add_argument("--val_anno_dir", type=str,
                        default='../data/instance-level_human_parsing/Validation/Human_ids',
                        help="val human mask path")
    return parser.parse_args()


def main(args):
    INFO = {
        "description": args.split_name + " Dataset",
        "url": "",
        "version": "",
        "year": 2019,
        "contributor": "xyq",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "",
            "url": ""
        }
    ]

    CATEGORIES = [
        {
            'id': 1,
            'name': 'person',
            'supercategory': 'person',
        },
    ]

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1

    for image_name in os.listdir(args.train_img_dir):
        image = Image.open(os.path.join(args.train_img_dir, image_name))
        image_info = pycococreatortools.create_image_info(
            image_id, image_name, image.size
        )
        coco_output["images"].append(image_info)

        human_mask_name = os.path.splitext(image_name)[0] + '.png'
        human_mask = np.asarray(Image.open(os.path.join(args.train_anno_dir, human_mask_name)))
        human_gt_labels = np.unique(human_mask)

        for i in range(1, len(human_gt_labels)):
            category_info = {'id': 1, 'is_crowd': 0}
            binary_mask = np.uint8(human_mask == i)
            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=10
            )
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            segmentation_id += 1
        image_id += 1

    if not os.path.exists(args.json_save_dir):
        os.makedirs(args.json_save_dir)
    if not args.use_val:
        with open('{}/{}_train.json'.format(args.json_save_dir, args.split_name), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)
    else:
        for image_name in os.listdir(args.val_img_dir):
            image = Image.open(os.path.join(args.val_img_dir, image_name))
            image_info = pycococreatortools.create_image_info(
                image_id, image_name, image.size
            )
            coco_output["images"].append(image_info)

            human_mask_name = os.path.splitext(image_name)[0] + '.png'
            human_mask = np.asarray(Image.open(os.path.join(args.val_anno_dir, human_mask_name)))
            human_gt_labels = np.unique(human_mask)

            for i in range(1, len(human_gt_labels)):
                category_info = {'id': 1, 'is_crowd': 0}
                binary_mask = np.uint8(human_mask == i)
                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=10
                )
                if annotation_info is not None:
                    coco_output["annotations"].append(annotation_info)

                segmentation_id += 1
            image_id += 1

        with open('{}/{}_trainval.json'.format(args.json_save_dir, args.split_name), 'w') as output_json_file:
            json.dump(coco_output, output_json_file)

    coco_output_val = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id_val = 1
    segmentation_id_val = 1

    for image_name in os.listdir(args.val_img_dir):
        image = Image.open(os.path.join(args.val_img_dir, image_name))
        image_info = pycococreatortools.create_image_info(
            image_id_val, image_name, image.size
        )
        coco_output_val["images"].append(image_info)

        human_mask_name = os.path.splitext(image_name)[0] + '.png'
        human_mask = np.asarray(Image.open(os.path.join(args.val_anno_dir, human_mask_name)))
        human_gt_labels = np.unique(human_mask)

        for i in range(1, len(human_gt_labels)):
            category_info = {'id': 1, 'is_crowd': 0}
            binary_mask = np.uint8(human_mask == i)
            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id_val, image_id_val, category_info, binary_mask,
                image.size, tolerance=10
            )
            if annotation_info is not None:
                coco_output_val["annotations"].append(annotation_info)

            segmentation_id_val += 1
        image_id_val += 1

    with open('{}/{}_val.json'.format(args.json_save_dir, args.split_name), 'w') as output_json_file_val:
        json.dump(coco_output_val, output_json_file_val)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
