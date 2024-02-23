import numpy as np
import cv2, torch
import os
import json
import argparse
import pycocotools.mask as mask_util
from tqdm import tqdm


def bbox_expand(img_height, img_width, bbox, exp_ratio):
    x_min, y_min, x_max, y_max = bbox[:]
    exp_x = (x_max - x_min) * ((exp_ratio - 1) / 2)
    exp_y = (y_max - y_min) * ((exp_ratio - 1) / 2)
    new_x_min = 0 if x_min - exp_x < 0 else np.round(x_min - exp_x)
    new_y_min = 0 if y_min - exp_y < 0 else np.round(y_min - exp_y)
    new_x_max = img_width - 1 if x_max + exp_x > img_width - 1 else np.round(x_max + exp_x)
    new_y_max = img_height - 1 if y_max + exp_y > img_height - 1 else np.round(y_max + exp_y)
    return int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)


def make_crop_and_mask(img_info, pred, file_list, crop_save_dir, mask_save_dir, args):
    img_name = img_info['file_name']
    img_id = img_info['id'] - 1  # img_info['id'] start form 1
    img_w = img_info['width']
    img_h = img_info['height']

    img = cv2.imread(os.path.join(args.img_dir, img_name))

    exp_bbox = []
    ori_bbox = []
    bbox_name_list = []
    bbox_score_list = []
    person_idx = 0

    panoptic_seg = np.zeros((img_h, img_w), dtype=np.uint8)
    assert len(pred[img_id]['instances']) > 0, 'image without instance prediction'

    for instance in pred[img_id]['instances']:
        score = instance['score']
        if score < args.conf_thres:
            break

        mask = mask_util.decode(instance['segmentation'])
        mask_area = mask.sum()

        if mask_area == 0:  # if mask_area < img_w*img_h/1000:
            continue

        intersect = (mask > 0) & (panoptic_seg > 0)
        intersect_area = intersect.sum()

        if intersect_area * 1.0 / mask_area > args.overlap_threshold:  # todo add args
            continue

        if intersect_area > 0:
            mask = mask & (panoptic_seg == 0)

        person_idx += 1
        panoptic_seg = np.where(mask == 0, panoptic_seg, person_idx)

        bbox_score_list.append(score)

        ins_bbox = instance['bbox']  # [x,y,w,h] format
        x_min, y_min, box_w, box_h = ins_bbox
        x_max, y_max = x_min + box_w, y_min + box_h
        exp_x_min, exp_y_min, exp_x_max, exp_y_max = bbox_expand(img_h, img_w, [x_min, y_min, x_max, y_max],
                                                                 args.exp_ratio)
        crop_img = img[exp_y_min:exp_y_max + 1, exp_x_min:exp_x_max + 1, :]
        exp_bbox.append([exp_x_min, exp_y_min, exp_x_max, exp_y_max])
        ori_bbox.append([x_min, y_min, x_max, y_max])
        bbox_name = os.path.splitext(img_name)[0] + '_' + str(person_idx) + '_msrcnn.jpg'
        bbox_name_list.append(bbox_name)

        cv2.imwrite(os.path.join(crop_save_dir, bbox_name), crop_img)

    assert person_idx > 0, 'image without instance'
    mask_name = os.path.splitext(img_name)[0] + '_mask.npy'
    np.save(os.path.join(mask_save_dir, mask_name), panoptic_seg)

    ############## json writing ##################
    item = {}
    item['dataset'] = 'CIHP'
    item['im_name'] = img_name
    item['img_height'] = img_h
    item['img_width'] = img_w
    item['center'] = [img_h / 2, img_w / 2]
    item['person_num'] = person_idx
    item['person_bbox'] = exp_bbox
    item['real_person_bbox'] = ori_bbox
    item['person_bbox_score'] = bbox_score_list
    item['bbox_name'] = bbox_name_list
    item['mask_name'] = mask_name
    file_list.append(item)
    json_file = {'root': file_list}
    return json_file, file_list


def get_arguments():
    parser = argparse.ArgumentParser(description="crop person val/test demo for inference")
    parser.add_argument("--exp_ratio", type=float, default=1.2)
    parser.add_argument("--overlap_threshold", type=float, default=0.5)
    parser.add_argument("--conf_thres", type=float, default=0.5)
    parser.add_argument("--img_dir", type=str,
                        default='/data03/v_xuyunqiu/data/instance-level_human_parsing/Testing/Images')
    parser.add_argument("--save_dir", type=str,
                        default='/data03/v_xuyunqiu/Projects/experiment_data/testing/resnest_200_TTA_mask_nms_all_data')
    parser.add_argument("--img_list", type=str,
                        default='/data03/v_xuyunqiu/Projects/pycococreator/annotations/CIHP_test.json')
    parser.add_argument("--det_res", type=str,
                        default='/data02/v_xuyunqiu/detectron2-ResNeSt/tools/output_cihp_inference_resnest/inference_TTA/instances_predictions.pth')
    return parser.parse_args()


def main(args):
    img_info_list = json.load(open(args.img_list, encoding='UTF-8'))
    pred = torch.load(args.det_res)

    crop_save_dir = os.path.join(args.save_dir, 'crop_pic')
    if not os.path.exists(crop_save_dir):
        os.makedirs(crop_save_dir)
    mask_save_dir = os.path.join(args.save_dir, 'crop_mask')
    if not os.path.exists(mask_save_dir):
        os.makedirs(mask_save_dir)

    file_list = []
    for img_info in tqdm(img_info_list['images']):
        json_file, file_list = make_crop_and_mask(img_info, pred, file_list, crop_save_dir, mask_save_dir, args)
        with open(os.path.join(args.save_dir, 'crop.json'), 'w') as f:
            json.dump(json_file, f, indent=2)


if __name__ == '__main__':
    args = get_arguments()
    main(args)
