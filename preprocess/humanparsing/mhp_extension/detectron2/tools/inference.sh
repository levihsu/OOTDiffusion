python finetune_net.py \
	--num-gpus 1 \
	--config-file ../configs/Misc/parsing_inference.yaml \
	--eval-only MODEL.WEIGHTS ./model_final.pth TEST.AUG.ENABLED False 
