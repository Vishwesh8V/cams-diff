"""
Train a diffusion model on images.
"""

import argparse
import json, torch, os
from cams_generation import gaussian_diffusion as gd
from cams_generation.respace import SpacedDiffusion, space_timesteps
import numpy as np
from cams_generation import dist_util, logger
from cams_generation.transformer_model2 import TransformerNetModel2
from cams_generation.image_datasets import load_data
from cams_generation.text_datasets import load_data_text
from cams_generation.resample import create_named_schedule_sampler
from cams_generation.script_util import (
    model_and_diffusion_defaults,
    add_dict_to_argparser,
)
from transformers import AutoTokenizer
from cams_generation.train_util import TrainLoop
from transformers import set_seed
from functools import partial
from cams_generation.test_util import get_weights, compute_logp
from cams_generation.rounding import load_models, load_tokenizer
import torch.distributed as dist
import wandb
from mytokenizers import SimpleSmilesTokenizer,regexTokenizer
from mydatasets import get_dataloader,ChEBIdataset
import warnings
import torch.multiprocessing as mp
warnings.filterwarnings("ignore")

def create_argparser():
    defaults = dict()
    text_defaults = dict(
        attention_resolutions='16,8', 
        batch_size=64, 
        cache_mode='no', 
        checkpoint_path='../../checkpoints_new', 
        class_cond=False, 
        commonGen_train='diffusion_lm/common-gen/commongen_data', 
        config='ll', 
        config_name='bert-base-uncased', 
        data_dir='../../datasets/SMILES/',
        train_split='train_val_256',
        dataset_config_name='wikitext-2-raw-v1', 
        dataset_name='wikitext', 
        diffusion_steps=2000, 
        dropout=0.1, 
        e2e_train='', 
        ema_rate='0.9999', 
        emb_scale_factor=1.0, 
        eval_interval=2000, 
        experiment='random', 
        experiment_mode='lm', 
        fp16_scale_growth=0.001, 
        gradient_clipping=2.4, 
        image_size=8, 
        in_channel=16, 
        learn_sigma=False, 
        log_interval=20, 
        logits_mode=1, 
        lr = 0.00005,
        # lr=0.0001, # Lower the learing rate in 2024/5/5 for better convergence 
        lr_anneal_steps=200000, 
        microbatch=-1, 
        modality='e2e-tgt', 
        model_arch='transformer', 
        model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None', 
        noise_level=0.0, 
        noise_schedule='sqrt', 
        num_channels=128, 
        num_heads=4, 
        num_heads_upsample=-1, 
        num_res_blocks=2, 
        out_channel=16, 
        padding_mode='pad', 
        predict_xstart=True, 
        preprocessing_num_workers=1, 
        rescale_learned_sigmas=True, 
        rescale_timesteps=True, 
        resume_checkpoint='', 
        roc_train='diffusion_lm/ROCstory', 
        save_interval=10000, 
        schedule_sampler='uniform', 
        seed=19991009, 
        sigma_small=False, 
        timestep_respacing='', 
        training_mode='e2e', 
        use_bert_tokenizer='no', 
        use_checkpoint=False, 
        use_fp16=False, 
        use_kl=False, 
        use_scale_shift_norm=True, 
        weight_decay=0.0,
        model_in_channels = 32,
        model_model_channels = 128,
        model_dropout = 0.1,
        model_hidden_size = 1024,
        model_num_attention_heads = 16,
        model_num_hidden_layers = 12,
        learned_mean_embed=False,
        denoise=False,
        denoise_rate=0.2,
        reg_rate=0.0,
        adaptive_noise=False,
        token_max_length=256,
        pad_tok_id=0,
        loss_update_granu=50,
        schedule_update_stride=500,
        save_dir='',
        adaptive_schedule_path='',
        resume_warmup_steps=300,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    import os
    os.environ['CUDA_DEVICES_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    world_size=1
    mp.spawn(main_worker,args=(world_size,),nprocs=world_size,join=True)
