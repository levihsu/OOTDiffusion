import pdb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datasets.simple_extractor_dataset import SimpleFolderDataset
from utils.transforms import transform_logits
from tqdm import tqdm
from PIL import Image


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def delete_irregular(logits_result):
    parsing_result = np.argmax(logits_result, axis=2)
    upper_cloth = np.where(parsing_result == 4, 255, 0)
    contours, hierarchy = cv2.findContours(upper_cloth.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        area.append(abs(a))
    if len(area) != 0:
        top = area.index(max(area))
        M = cv2.moments(contours[top])
        cY = int(M["m01"] / M["m00"])

    dresses = np.where(parsing_result == 7, 255, 0)
    contours_dress, hierarchy_dress = cv2.findContours(dresses.astype(np.uint8),
                                                       cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area_dress = []
    for j in range(len(contours_dress)):
        a_d = cv2.contourArea(contours_dress[j], True)
        area_dress.append(abs(a_d))
    if len(area_dress) != 0:
        top_dress = area_dress.index(max(area_dress))
        M_dress = cv2.moments(contours_dress[top_dress])
        cY_dress = int(M_dress["m01"] / M_dress["m00"])
    wear_type = "dresses"
    if len(area) != 0:
        if len(area_dress) != 0 and cY_dress > cY:
            irregular_list = np.array([4, 5, 6])
            logits_result[:, :, irregular_list] = -1
        else:
            irregular_list = np.array([5, 6, 7, 8, 9, 10, 12, 13])
            logits_result[:cY, :, irregular_list] = -1
            wear_type = "cloth_pant"
        parsing_result = np.argmax(logits_result, axis=2)
    # pad border
    parsing_result = np.pad(parsing_result, pad_width=1, mode='constant', constant_values=0)
    return parsing_result, wear_type



def hole_fill(img):
    img_copy = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(img, mask, (0, 0), 255)
    img_inverse = cv2.bitwise_not(img)
    dst = cv2.bitwise_or(img_copy, img_inverse)
    return dst

def refine_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    area = []
    for j in range(len(contours)):
        a_d = cv2.contourArea(contours[j], True)
        area.append(abs(a_d))
    refine_mask = np.zeros_like(mask).astype(np.uint8)
    if len(area) != 0:
        i = area.index(max(area))
        cv2.drawContours(refine_mask, contours, i, color=255, thickness=-1)
        # keep large area in skin case
        for j in range(len(area)):
          if j != i and area[i] > 2000:
             cv2.drawContours(refine_mask, contours, j, color=255, thickness=-1)
    return refine_mask

def refine_hole(parsing_result_filled, parsing_result, arm_mask):
    filled_hole = cv2.bitwise_and(np.where(parsing_result_filled == 4, 255, 0),
                                  np.where(parsing_result != 4, 255, 0)) - arm_mask * 255
    contours, hierarchy = cv2.findContours(filled_hole, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    refine_hole_mask = np.zeros_like(parsing_result).astype(np.uint8)
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], True)
        # keep hole > 2000 pixels
        if abs(a) > 2000:
            cv2.drawContours(refine_hole_mask, contours, i, color=255, thickness=-1)
    return refine_hole_mask + arm_mask

def onnx_inference(session, lip_session, input_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])
    dataset = SimpleFolderDataset(root=input_dir, input_size=[512, 512], transform=transform)
    dataloader = DataLoader(dataset)
    with torch.no_grad():
        for _, batch in enumerate(tqdm(dataloader)):
            image, meta = batch
            c = meta['center'].numpy()[0]
            s = meta['scale'].numpy()[0]
            w = meta['width'].numpy()[0]
            h = meta['height'].numpy()[0]
            output = session.run(None, {"input.1": image.numpy().astype(np.float32)})
            upsample = torch.nn.Upsample(size=[512, 512], mode='bilinear', align_corners=True)
            upsample_output = upsample(torch.from_numpy(output[1][0]).unsqueeze(0))
            upsample_output = upsample_output.squeeze()
            upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC
            logits_result = transform_logits(upsample_output.data.cpu().numpy(), c, s, w, h, input_size=[512, 512])
            parsing_result = np.argmax(logits_result, axis=2)
            parsing_result = np.pad(parsing_result, pad_width=1, mode='constant', constant_values=0)
            # try holefilling the clothes part
            arm_mask = (parsing_result == 14).astype(np.float32) \
                       + (parsing_result == 15).astype(np.float32)
            upper_cloth_mask = (parsing_result == 4).astype(np.float32) + arm_mask
            img = np.where(upper_cloth_mask, 255, 0)
            dst = hole_fill(img.astype(np.uint8))
            parsing_result_filled = dst / 255 * 4
            parsing_result_woarm = np.where(parsing_result_filled == 4, parsing_result_filled, parsing_result)
            # add back arm and refined hole between arm and cloth
            refine_hole_mask = refine_hole(parsing_result_filled.astype(np.uint8), parsing_result.astype(np.uint8),
                                           arm_mask.astype(np.uint8))
            parsing_result = np.where(refine_hole_mask, parsing_result, parsing_result_woarm)
            # remove padding
            parsing_result = parsing_result[1:-1, 1:-1]

        dataset_lip = SimpleFolderDataset(root=input_dir, input_size=[473, 473], transform=transform)
        dataloader_lip = DataLoader(dataset_lip)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader_lip)):
                image, meta = batch
                c = meta['center'].numpy()[0]
                s = meta['scale'].numpy()[0]
                w = meta['width'].numpy()[0]
                h = meta['height'].numpy()[0]

                output_lip = lip_session.run(None, {"input.1": image.numpy().astype(np.float32)})
                upsample = torch.nn.Upsample(size=[473, 473], mode='bilinear', align_corners=True)
                upsample_output_lip = upsample(torch.from_numpy(output_lip[1][0]).unsqueeze(0))
                upsample_output_lip = upsample_output_lip.squeeze()
                upsample_output_lip = upsample_output_lip.permute(1, 2, 0)  # CHW -> HWC
                logits_result_lip = transform_logits(upsample_output_lip.data.cpu().numpy(), c, s, w, h,
                                                     input_size=[473, 473])
                parsing_result_lip = np.argmax(logits_result_lip, axis=2)
    # add neck parsing result
    neck_mask = np.logical_and(np.logical_not((parsing_result_lip == 13).astype(np.float32)),
                               (parsing_result == 11).astype(np.float32))
    parsing_result = np.where(neck_mask, 18, parsing_result)
    palette = get_palette(19)
    output_img = Image.fromarray(np.asarray(parsing_result, dtype=np.uint8))
    output_img.putpalette(palette)
    face_mask = torch.from_numpy((parsing_result == 11).astype(np.float32))

    return output_img, face_mask



