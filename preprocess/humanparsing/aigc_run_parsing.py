import pdb
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import load_atr_model, load_lip_model, inference
import torch


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        self.atr_model = load_atr_model()
        self.lip_model = load_lip_model()
        

    def __call__(self, input_image):
        torch.cuda.set_device(self.gpu_id)
        parsed_image, face_mask = inference(self.atr_model, self.lip_model, input_image)
        return parsed_image, face_mask
