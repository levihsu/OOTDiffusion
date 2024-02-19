python ./coco_style_annotation_creator/human_to_coco.py \
    --dataset 'CIHP' \
    --json_save_dir './data/CIHP/annotations' \
    --train_img_dir './data/CIHP/Training/Images' \
    --train_anno_dir './data/CIHP/Training/Human_ids' \
    --val_img_dir './data/CIHP/Validation/Images' \
    --val_anno_dir './data/CIHP/Validation/Human_ids'


python ./coco_style_annotation_creator/test_human2coco_format.py \
    --dataset 'CIHP' \
    --json_save_dir './data/CIHP/annotations' \
    --test_img_dir './data/CIHP/Testing/Images'

