from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc,
    test_batch_prediction_acc,
    test_prediction_acc,
    test_generation_quality,
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)

def compute_safety_edit_quality(
    dscd,
    save_path,
    model,
    # model_name,
    # hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    toxic_layer: int,
    generate_kwargs:typing.Dict,

    # test_generation = False
    max_tokens = 1024, 
    max_output_tokens: int = 500,
    


) -> typing.Dict:
    # print(record['general_prompt'])
    # print(record["prompt"])
    batch = [record["prompt"]] + record['general_prompt']
    # print("batch:"+str(batch))
    # #def test_safety_gen(
    #     model, 
    #     tokenizer, 
    #     test_prompt, 
    #     cuda,
    #     max_tokens = 1624, 
    #     max_output_tokens=600,
    #     max_new_tokens=256, 
    #     top_p=0.95,
        
    #     top_k=0,
    #     temperature=0.8,
    #     mature_layer=None,
    #     premature_layer=None,
    #     toxic_layer = None,
    #     candidate_premature_layers=[],
    #     mode='dola', 
    #     verbose=True, 
    #     remove_stop_words=False,
    #     relative_top=0.1, 
    #     **kwargs):
    DS, DG_onlyQ, DG_otherA, DG_otherQ, DG_otherAQ = test_safety_gen(dscd,save_path,model, tok, batch, device, max_tokens, max_output_tokens,toxic_layer, **generate_kwargs)
    ret = {
        "DS": DS,
        "DG_onlyQ": DG_onlyQ,
        "DG_otherA": DG_otherA,
        "DG_otherQ": DG_otherQ,
        "DG_otherAQ": DG_otherAQ
    }
    return ret

def ccks_compute_safety_edit_quality(
    model,
    # model_name,
    # hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    # test_generation = False
    max_tokens = 500,
    max_output_tokens: int = 400,
) -> typing.Dict:
    batch = [record["prompt"]] + record['general_prompt']
    DS, DG_otherAQ = test_safety_gen(model, tok, batch, device, max_tokens, max_output_tokens)
    ret = {
        "DS": DS,
        "DG_otherAQ": DG_otherAQ
    }
    return ret
