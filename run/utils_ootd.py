import pdb

import numpy as np
import cv2
from PIL import Image, ImageDraw

label_map = {
    "background": 0,
    "hat": 1,
    "hair": 2,
    "sunglasses": 3,
    "upper_clothes": 4,
    "skirt": 5,
    "pants": 6,
    "dress": 7,
    "belt": 8,
    "left_shoe": 9,
    "right_shoe": 10,
    "head": 11,
    "left_leg": 12,
    "right_leg": 13,
    "left_arm": 14,
    "right_arm": 15,
    "bag": 16,
    "scarf": 17,
}

def extend_arm_mask(wrist, elbow, scale):
  wrist = elbow + scale * (wrist - elbow)
  return wrist

def hole_fill(img):
    img = np.pad(img[1:-1, 1:-1], pad_width = 1, mode = 'constant', constant_values=0)
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

    return refine_mask

def get_mask_location(model_type, category, model_parse: Image.Image, keypoint: dict, width=384,height=512):
    im_parse = model_parse.resize((width, height), Image.NEAREST)
    parse_array = np.array(im_parse)

    if model_type == 'hd':
        arm_width = 60
    elif model_type == 'dc':
        arm_width = 45
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")

    parse_head = (parse_array == 1).astype(np.float32) + \
                 (parse_array == 3).astype(np.float32) + \
                 (parse_array == 11).astype(np.float32)

    parser_mask_fixed = (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                        (parse_array == label_map["hat"]).astype(np.float32) + \
                        (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                        (parse_array == label_map["bag"]).astype(np.float32)

    # (parse_array == label_map["scarf"]).astype(np.float32) + \

    parser_mask_changeable = (parse_array == label_map["background"]).astype(np.float32)

    arms_left = (parse_array == 14).astype(np.float32)
    arms_right = (parse_array == 15).astype(np.float32)
    arms = arms_left + arms_right

    if category == 'dresses':
        # 裙子加腿 mask
        parse_mask = (parse_array == 7).astype(np.float32) + \
                     (parse_array == 4).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32) + \
                     (parse_array == 6).astype(np.float32)
                     # (parse_array == 12).astype(np.float32) + \
                     # (parse_array == 13).astype(np.float32) + \

        # 除fixed的区域，都是parser_mask_changeable
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))

    elif category == 'upper_body':
        # 上衣mask
        parse_mask = (parse_array == 4).astype(np.float32) + (parse_array == 7).astype(np.float32)
        # 下半身衣物fixed
        parser_mask_fixed_lower_cloth = (parse_array == label_map["skirt"]).astype(np.float32) + \
                                        (parse_array == label_map["pants"]).astype(np.float32)
        # parser_mask_fixed_lower_cloth = cv2.erode(parser_mask_fixed_lower_cloth, np.ones((5, 5), np.uint16))
        parser_mask_fixed += parser_mask_fixed_lower_cloth
        # 除fixed的区域，都是parser_mask_changeable
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    elif category == 'lower_body':
        # 裤子加腿 mask
        parse_mask = (parse_array == 6).astype(np.float32) + \
                     (parse_array == 12).astype(np.float32) + \
                     (parse_array == 13).astype(np.float32) + \
                     (parse_array == 5).astype(np.float32)
        # 手臂fixed
        parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                             (parse_array == 14).astype(np.float32) + \
                             (parse_array == 15).astype(np.float32)
        # 除fixed的区域，都是parser_mask_changeable
        parser_mask_changeable += np.logical_and(parse_array, np.logical_not(parser_mask_fixed))
    else:
        raise NotImplementedError

    # Load pose points
    pose_data = keypoint["pose_keypoints_2d"]
    pose_data = np.array(pose_data)
    pose_data = pose_data.reshape((-1, 2))

    im_arms_left = Image.new('L', (width, height))
    im_arms_right = Image.new('L', (width, height))
    arms_draw_left = ImageDraw.Draw(im_arms_left)
    arms_draw_right = ImageDraw.Draw(im_arms_right)
    if category == 'dresses' or category == 'upper_body':
        shoulder_right = np.multiply(tuple(pose_data[2][:2]), height / 512.0)
        shoulder_left = np.multiply(tuple(pose_data[5][:2]), height / 512.0)
        elbow_right = np.multiply(tuple(pose_data[3][:2]), height / 512.0)
        elbow_left = np.multiply(tuple(pose_data[6][:2]), height / 512.0)
        wrist_right = np.multiply(tuple(pose_data[4][:2]), height / 512.0)
        wrist_left = np.multiply(tuple(pose_data[7][:2]), height / 512.0)
        ARM_LINE_WIDTH = int(arm_width / 512 * height)
        size_left = [shoulder_left[0] - ARM_LINE_WIDTH // 2, shoulder_left[1] - ARM_LINE_WIDTH // 2, shoulder_left[0] + ARM_LINE_WIDTH // 2, shoulder_left[1] + ARM_LINE_WIDTH // 2]
        size_right = [shoulder_right[0] - ARM_LINE_WIDTH // 2, shoulder_right[1] - ARM_LINE_WIDTH // 2, shoulder_right[0] + ARM_LINE_WIDTH // 2,
                      shoulder_right[1] + ARM_LINE_WIDTH // 2]
        

        if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
            if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                # arms_draw_left.line(
                #     np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white',
                #     60, 'curve')
                # arms_draw_left.arc(size_left, 0, 360, 'white', 30)
                im_arms_right = arms_right
            else:
                # arms_draw_left.line(
                #     np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white',
                #     60, 'curve')
                # arms_draw_left.arc(size_left, 0, 360, 'white', 30)
                im_arms_right = arms_right
                # arms_draw_right.line(np.concatenate((shoulder_right, elbow_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                # arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH//2)
        else:
            wrist_right = extend_arm_mask(wrist_right, elbow_right, 1.2)
            arms_draw_right.line(np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_right.arc(size_right, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        if wrist_left[0] <= 1. and wrist_left[1] <= 1.:
            if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                # arms_draw_right.line(
                #     np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(),
                #     'white', 60, 'curve')
                # arms_draw_right.arc(size_right, 0, 360, 'white', 30)
                im_arms_left = arms_left
            else:
                # arms_draw_left.line(np.concatenate((elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
                im_arms_left = arms_left
                # arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH//2)

                # arms_draw_right.line(
                #     np.concatenate((shoulder_right, elbow_right, wrist_right)).astype(np.uint16).tolist(),
                #     'white', 60, 'curve')
                # arms_draw_right.arc(size_right, 0, 360, 'white', 30)
        else:
            # extend arm scale 1.2
            wrist_left = extend_arm_mask(wrist_left, elbow_left, 1.2)
            arms_draw_left.line(np.concatenate((wrist_left, elbow_left, shoulder_left)).astype(np.uint16).tolist(), 'white', ARM_LINE_WIDTH, 'curve')
            arms_draw_left.arc(size_left, 0, 360, 'white', ARM_LINE_WIDTH // 2)

        # im_arms_left.save("leftarm.jpg")
        # im_arms_right.save("rightarm.jpg")
        hands_left = np.logical_and(np.logical_not(im_arms_left), arms_left)
        # hands_left_refine = refine_mask(hands_left)
        # im_arms_left += np.logical_and(np.logical_not(hands_left_refine), hands_left)
        # Image.fromarray(((hands_left > 0) * 127.5 + 127.5).astype(np.uint8)).save("hands_left.jpg")
        hands_right = np.logical_and(np.logical_not(im_arms_right), arms_right)
        # hands_right_refine = refine_mask(hands_right)
        # im_arms_right += np.logical_and(np.logical_not(hands_right_refine), hands_right)
        # Image.fromarray(((hands_right > 0) * 127.5 + 127.5).astype(np.uint8)).save("hands_right.jpg")
        # Image.fromarray(((parser_mask_fixed > 0) * 127.5 + 127.5).astype(np.uint8)).save("fixed_before.jpg")
        parser_mask_fixed += hands_left + hands_right
        # Image.fromarray(((parser_mask_fixed > 0) * 127.5 + 127.5).astype(np.uint8)).save("fixed_after.jpg")

    # parser_mask_fixed 加入脸部和双肩以上脖子
    parser_mask_fixed = np.logical_or(parser_mask_fixed, parse_head)
    parse_mask = cv2.dilate(parse_mask, np.ones((5, 5), np.uint16), iterations=5)
    if category == 'dresses' or category == 'upper_body':
        neck_mask = (parse_array == 18).astype(np.float32)
        # Image.fromarray(((parse_mask > 0) * 127.5 + 127.5).astype(np.uint8)).save("mask_before.jpg")
        neck_mask = cv2.dilate(neck_mask, np.ones((5, 5), np.uint16), iterations=1)
        neck_mask = np.logical_and(neck_mask, np.logical_not(parse_head))
        parse_mask = np.logical_or(parse_mask, neck_mask)
        # Image.fromarray(((parse_mask > 0) * 127.5 + 127.5).astype(np.uint8)).save("mask_after.jpg")
        # 单独控制手臂部分大小
        arm_mask = cv2.dilate(np.logical_or(im_arms_left, im_arms_right).astype('float32'), np.ones((5, 5), np.uint16), iterations=4)
        # Image.fromarray(((arm_mask > 0) * 127.5 + 127.5).astype(np.uint8)).save("arm_mask.jpg")
        parse_mask += np.logical_or(parse_mask, arm_mask)

    # pdb.set_trace()
    # 除了fix（hands排除）就是changeable，not parse_mask 除了要mask的 其余置1（除了fixed区域与mask区域，其余都为1）
    # Image.fromarray(((parse_mask > 0) * 127.5 + 127.5).astype(np.uint8)).save("mask_addarm.jpg")
    parse_mask = np.logical_and(parser_mask_changeable, np.logical_not(parse_mask))

    # parser_mask_fixed + parse_mask 结合，把fixed加入，只剩要mask区域（若有侵蚀到fixed的操作也会被fixed重新取回）
    # Image.fromarray(((parser_mask_fixed > 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)).save("mask_fixed.jpg")
    parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
    inpaint_mask = 1 - parse_mask_total
    img = np.where(inpaint_mask, 255, 0)
    dst = hole_fill(img.astype(np.uint8))
    dst = refine_mask(dst)
    inpaint_mask = dst / 255 * 1
    mask = Image.fromarray(inpaint_mask.astype(np.uint8) * 255)
    mask_gray = Image.fromarray(inpaint_mask.astype(np.uint8) * 127)

    return mask, mask_gray