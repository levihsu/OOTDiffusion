all: help

help:
	@echo ""
	@echo "   -- Help Menu: step by step"
	@echo ""
	@echo "   1. make build              - build a image version"
	@echo "   2. make test               - Test whether the built image can enable CUDA"
	@echo "   3. make deploy             - Deploy ootd project"
	@echo "   4. make exec               - Enter the ootd container(The container has already configured the software environment), please download all necessary models in the checkpoints folder by yourself, follow the Inference in the readme"
	@echo ""

username=st
BASE_IMAGE_NAME=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
NAME=ootd
VERSION=v0.0.2
CONTAINER_NAME=ootd
half:
	@python run_ootd.py --model_path ./examples/model/01008_00.jpg --cloth_path ./examples/garment/00055_00.jpg --scale 2.0 --sample 4
exec:
	@docker exec -it ${CONTAINER_NAME} /bin/bash

deploy:
	@docker run -itd --gpus all \
		--name ${CONTAINER_NAME} \
		-p 7865:7865 \
		-v .:/app \
		${NAME}:${VERSION} /bin/bash
test:
	@docker run -it --rm \
		--gpus all \
		--name test \
		${NAME}:${VERSION} nvidia-smi
build:
	@docker build -t ${NAME}:${VERSION} .

stop:
	@docker stop ${CONTAINER_NAME}
clean: stop
	@docker rm ${CONTAINER_NAME}

hello:
	@echo "makefile,$(username), hello world !"

