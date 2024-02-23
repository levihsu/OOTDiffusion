import os

DATASET = 'VIP'  # DATASET: MHPv2 or CIHP or VIP
TYPE = 'crop_pic'  # crop_pic or DemoDataset
IMG_DIR = '../demo/cropped_img/crop_pic'
SAVE_DIR = '../demo/cropped_img'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

with open(os.path.join(SAVE_DIR, TYPE + '.txt'), "w") as f:
    for img_name in os.listdir(IMG_DIR):
        f.write(img_name[:-4] + '\n')
