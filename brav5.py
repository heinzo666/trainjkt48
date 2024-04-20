#!/usr/bin/env python3
# import pipeline and scheduler from https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7/
from lcm_pipeline import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler
import torch

scheduler = LCMScheduler.from_pretrained("sinkinai/Beautiful-Realistic-Asians-v5", subfolder="scheduler")

pipe = LatentConsistencyModelPipeline.from_pretrained("sinkinai/Beautiful-Realistic-Asians-v5", scheduler=scheduler, dtype=torch.float16)


