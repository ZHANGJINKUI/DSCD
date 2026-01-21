import argparse
import time
import csv
import tqdm
import os
import json
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import argparse
import warnings
import pandas as pd
import numpy as np
from DINM.models import apply_dinm_to_model,DINMHyperParams
hparams_path="/mnt/sdb/zjk/EasyEdit/hparams/DINM/llama2-7b.yaml"#注释：祛毒编辑的模型参数设置

hparams = DINMHyperParams.from_hparams(hparams_path)
# def _locate_toxic_layer(model, tokenizer, requests, hparams, **kwargs):
#     toxic_layer = []
#     input = tokenizer(requests, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{hparams.device}") 
#     with torch.no_grad():
#         outputs = model(**input)
#         # 获取每一层的隐藏状态
#     hidden_states = outputs.hidden_states
#     for j in range(1):
#         max_distance_layer = None
#         max_distance_value = float('-inf')

#         for layer_index in range(1, len(hidden_states)):
#             euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[layer_index][j * 2 + 1], p=2)

#             if euclidean_distance.item() > max_distance_value:
#                 max_distance_value = euclidean_distance.item()
#                 max_distance_layer = layer_index
#         toxic_layer.append(max_distance_layer-1)
#     return toxic_layer
def _locate_toxic_layer(model, tokenizer, requests, hparams, **kwargs):
    toxic_layer = []
    input = tokenizer(requests, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{hparams.device}") 
    with torch.no_grad():
        outputs = model(**input)
        # 获取每一层的隐藏状态
    hidden_states = outputs.hidden_states

    for j in range(1):
        max_distance_layer = None
        max_distance_value = float('-inf')

        for layer_index in range(len(hidden_states)):
            # 比较所有层之间的距离
            for compare_index in range(len(hidden_states)):
                if layer_index != compare_index:
                    euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[compare_index][j * 2 + 1], p=2)

                    if euclidean_distance.item() > max_distance_value:
                        max_distance_value = euclidean_distance.item()
                        max_distance_layer = layer_index
        toxic_layer.append(max_distance_layer-1)
    return toxic_layer
class DSCD:
    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        # tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        # torch_dtype = torch.bfloat16
        # model = AutoModelForCausalLM.from_pretrained(hparams.model_name, output_hidden_states=True,trust_remote_code=True, device_map='auto')

        tok = AutoTokenizer.from_pretrained(hparams.model_name)
        tok.pad_token_id = tok.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(hparams.model_name,
            low_cpu_mem_usage=True, **kwargs, device_map='auto')
            
#########################修改
        # torch_dtype = torch.bfloat16
        # # torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32

        # model = AutoModelForCausalLM.from_pretrained(hparams.model_name, output_hidden_states=True, torch_dtype=torch_dtype, trust_remote_code=True, device_map='auto')
############################修改
        # if self.device == "cuda" and self.num_gpus == 1:
        #     model.cuda()
        
        return model, tok
    # def load_model(self, model_name):
    #     if self.device == "cuda":
    #         kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
    #         if self.num_gpus == "auto":
    #             kwargs["device_map"] = "auto"
    #         else:
    #             self.num_gpus = int(self.num_gpus)
    #             if self.num_gpus != 1:
    #                 kwargs.update({
    #                     "device_map": "auto",
    #                     "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
    #                 })
    #     elif self.device == "cpu":
    #         kwargs = {}
    #     else:
    #         raise ValueError(f"Invalid device: {self.device}")
        
    #     tok = AutoTokenizer.from_pretrained(hparams.model_name)
    #     tok.pad_token_id = tok.eos_token_id
    #     model = AutoModelForCausalLM.from_pretrained(model_name,
    #         low_cpu_mem_usage=True, **kwargs)

    #     if self.device == "cuda" and self.num_gpus == 1:
    #         model.cuda()
        
    #     return model, tok
    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, input_text,request, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, toxic_layer = None,candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():
            # input_text = input_text+' '+"If the user's request is reasonable, give the correct answer. Please give the right response:"
            
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            max_len = input_ids.shape[-1] + max_new_tokens

            if mode == 'baseline':
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True,                                     dscd_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            elif mode == 'dscd-static':
                assert mature_layer is not None, "mature_layer must be specified"
                assert premature_layer is not None, "premature_layer must be specified"
                outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True,                                     dscd_decoding=True,
                                    mature_layer=mature_layer, premature_layer=premature_layer,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, **kwargs)
            elif mode == 'dscd':
                assert mature_layer is not None, "mature_layer must be specified"
                assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                toxic_layer = _locate_toxic_layer(self.model, self.tokenizer, request, hparams)
                toxic_layer = toxic_layer[0]+1
                print("toxic_layer:{}".format(toxic_layer))
                # outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                #                         output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                #                         top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                #                         mature_layer=mature_layer, toxic_layer = toxic_layer[0],premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs,)
                eos_token_id = self.tokenizer.eos_token_id
                # kwargs = dict(do_sample=True, max_new_tokens=1024, repetition_penalty=1.2, mode=mode, remove_stop_words=True, eos_token_id=eos_token_id)

                outputs = self.model.generate(
                    input_ids,
                    max_length=max_len,
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                                        dscd_decoding=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    stopping_criteria=self.stopping_criteria,
                    relative_top=relative_top,
                    mature_layer=mature_layer,
                    toxic_layer=toxic_layer,
                    premature_layer=None,
                    candidate_premature_layers=candidate_premature_layers,
                    eos_token_id=eos_token_id,
                    no_repeat_ngram_size=2,
                    **kwargs
                )               
                premature_layer_dist = outputs.premature_layer_dist
                presafe_layer_dist = outputs.presafe_layer_dist
            sequences, scores = outputs.sequences, outputs.scores

            # skip the tokens in the input prompt
            gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str, (premature_layer_dist if mode == 'dscd' else None)

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.1, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    # def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top: float = 0.999, min_tokens_to_keep: int = 1):
    #     scores_normalized = scores.log_softmax(dim=-1)
        
    #     # Calculate the number of top-k elements to select
    #     k = max(int(relative_top * scores.size(-1)), min_tokens_to_keep)
    #     print(f"Calculated k: {k}, based on relative_top: {relative_top} and scores.size(-1): {scores.size(-1)}")  # Debugging line
        
    #     # Get indices of the top-k maximum values
    #     topk_values, topk_indices = torch.topk(scores_normalized, k, dim=-1)
        
    #     # Create a mask with all True values
    #     mask = torch.ones_like(scores_normalized, dtype=torch.bool)
        
    #     # Set mask to False at topk_indices, indicating selected elements
    #     mask.scatter_(-1, topk_indices, False)
        
    #     return mask
    def lm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):

        with torch.no_grad():
            input_text = input_text1 + input_text2
            input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            if mode == 'baseline':
                outputs = self.model(input_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)  # logits to log probs

                # skip tokens in the prompt -- we only care about the answer
                outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

                # get logprobs for each token in the answer
                log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
                
            elif mode == 'dscd-static':
                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=[premature_layer, mature_layer],
                )

                assert premature_layer is not None
                base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)
                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

            elif mode == 'dscd':
                premature_layer_dist = {l:0 for l in candidate_premature_layers}
                picked_logits = []
                result_dict = {}
                premature_layers = []

                dict_outputs, outputs = self.model(
                    input_ids=input_ids,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    early_exit_layers=candidate_premature_layers + [mature_layer],
                )

                for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):
                    # Pick the less like layer to contrast with
                    # 1. Stacking all premature_layers into a new dimension
                    stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                    # 2. Calculate the softmax values for mature_layer and all premature_layers
                    softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 3. Calculate M, the average distribution
                    M = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)

                    # 4. Calculate log-softmax for the KL divergence
                    log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                    log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)

                    # 5. Calculate the KL divergences and then the JS divergences
                    kl1 = F.kl_div(log_softmax_mature_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    kl2 = F.kl_div(log_softmax_premature_layers, M, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                    js_divs = 0.5 * (kl1 + kl2)  # shape: (num_premature_layers, batch_size)

                    # 6. Reduce the batchmean
                    js_divs = js_divs.mean(-1)  # shape: (num_premature_layers,)
                    premature_layer = candidate_premature_layers[int(js_divs.argmax().cpu().item())]
                    premature_layer_dist[premature_layer] += 1

                    premature_layers.append(premature_layer)

                base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
                for i, l in enumerate(premature_layers):
                   base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
                final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
                final_logits = final_logits.log_softmax(dim=-1)
                base_logits = base_logits.log_softmax(dim=-1)
                diff_logits = final_logits - base_logits
                if post_softmax:
                    diff_logits = diff_logits.log_softmax(dim=-1)

                if relative_top > 0.0:
                    relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                    diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                    true_indices = torch.nonzero(relative_top_mask, as_tuple=False)

                    # 输出前 10% 的索引
                    # print("True indices (top 10%):", true_indices)

                    # 验证选中的数量是否为 10%
                    total_elements = final_logits.numel()
                    selected_elements = true_indices.size(0)
                    # print("Total elements:", total_elements)
                    # print("Selected elements:", selected_elements)
                    print("Percentage of selected elements:", selected_elements / total_elements * 100, "%")
                log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

                return log_probs, (premature_layer_dist if mode == 'dscd' else None)
    
    def Logits_D(self, input_text1, input_text2,toxic_layer, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):

        with torch.no_grad():

            input_text = input_text1 + input_text2


            input_ids = self.tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(self.device)#上下文和希望生成的安全答案
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt", padding=True).input_ids.to(self.device)#上下文,即prefix

            # input_ids = dict(self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device))#上下文和希望生成的安全答案
            input_text2_ids = self.tokenizer(input_text2, return_tensors="pt", padding=True).to(self.device).input_ids.to(self.device)#上下文和希望生成的安全答案
            # prefix_ids = self.tokenizer(input_text1, return_tensors="pt", padding=True).input_ids.to(self.device)#上下文,即prefix
            # input_ids = prefix_ids
            out_labels =input_text2_ids.masked_fill(input_text2_ids == self.tokenizer.pad_token_id, -1000)

            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            # continue_ids = input_ids[0,:]

                
            # premature_layer_dist = {l:0 for l in candidate_premature_layers}
            safe_layers_dist =candidate_premature_layers+[mature_layer]
            safe_layer_dist = {l:0 for l in safe_layers_dist}
            premature_layer_dist = {l:0 for l in candidate_premature_layers}


            picked_logits = []
            result_dict = {}
            premature_layers = []
            safe_layers = []
            
            dict_outputs, outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                early_exit_layers=candidate_premature_layers+[mature_layer],
                
            )

            # for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):#处理了希望生成的安全答案的token,比较了每个token其成熟层与其差异最大的过早层之间的logits
            # for seq_i in range(1, input_ids.shape[-1] - 1):#处理了希望生成的安全答案的token,比较了每个token其成熟层与其差异最大的过早层之间的logits
            for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):

                # Pick the less like layer to contrast with
                # 1. Stacking all presafe_layers into a new dimension

                stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)

                stacked_presafe_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in safe_layers_dist], dim=0)#将候选过早层所有的隐藏状态叠加到第0维
                number_of_layers = len(candidate_premature_layers)  # 使用len()函数获取列表长度

                # print(number_of_layers) 
                num_iterations = input_ids.shape[-1] - 2  # 减去2是因为range的结束索引是倒数第二个
                num = number_of_layers*num_iterations


                # print("stacked_presafe_layers："+str(stacked_presafe_layers))#调试，过早层的选择！
                # 2. Calculate the softmax values for mature_layer and all presafe_layers
                # softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                ##########修改8.12,有毒层的加入
                softmax_toxic_layer = F.softmax(dict_outputs[toxic_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)

                softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)

                
                softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_presafe_layers, batch_size, num_features)

                softmax_presafe_layers = F.softmax(stacked_presafe_layers, dim=-1)  # shape: (num_presafe_layers, batch_size, num_features)

                # 3. Calculate M, the average distribution
                M = 0.5 * (softmax_toxic_layer[None, :, :]+ softmax_presafe_layers)  # shape: (num_presafe_layers, batch_size, num_features)
                N = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)


                # 4. Calculate log-softmax for the KL divergence
                log_softmax_toxic_layer = F.log_softmax(dict_outputs[toxic_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                # log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                log_softmax_presafe_layers = F.log_softmax(stacked_presafe_layers, dim=-1)  # shape: (num_presafe_layers, batch_size, num_features)
#############################
                log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)
                # 5. Calculate the KL divergences and then the JS divergences
                kl3 = F.kl_div(log_softmax_mature_layer[None, :, :], N, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                kl4 = F.kl_div(log_softmax_premature_layers, N, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                js_divs_mature = 0.5 * (kl3 + kl4)  # shape: (num_premature_layers, batch_size)
##########################
                # 5. Calculate the KL divergences and then the JS divergences
                kl1 = F.kl_div(log_softmax_toxic_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_presafe_layers, batch_size)，计算成熟层与M的kl散度
                kl2 = F.kl_div(log_softmax_presafe_layers, M, reduction='none').mean(-1)  # shape: (num_presafe_layers, batch_size)，计算过早层与M的kl散度，
                js_divs_safe = 0.5 * (kl1 + kl2)  # shape: (num_presafe_layers, batch_size)

                # 6. Reduce the batchmean
                js_divs_safe_mean = js_divs_safe.mean(-1)  # shape: (num_presafe_layers,)
                js_divs_safe_mean[0] = float('-inf')
                top_two_indices = js_divs_safe_mean.topk(2).indices.cpu().tolist()

                # 选择 safe_layer，确保不会选择第 0 层
                safe_layer = safe_layers_dist[top_two_indices[0]]
                # safe_layer = safe_layers_dist[int(js_divs_safe_mean.argmax().cpu().item())]
                safe_layer_dist[safe_layer] += 1
###############################################
                # 6. Reduce the batchmean
                js_divs_mature = js_divs_mature.mean(-1)  # shape: (num_premature_layers,)
                premature_layer = candidate_premature_layers[int(js_divs_mature.argmax().cpu().item())]
                premature_layer_dist[premature_layer] += 1

                premature_layers.append(premature_layer)
###########################################
                safe_layers.append(safe_layer)

            
            # for i, l in enumerate(safe_layers):

            #     safe_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]#将每一句话的logit值赋给 base_logits[i]，其中 i 是当前层在 presafe_layers 列表中的索引。
            
            #     # print("prefix_ids.shape[-1] - 1+i]:"+str(prefix_ids.shape[-1] - 1 + i))            safe_logits = torch.zeros_like(dict_outputs[toxic_layer][0, prefix_ids.shape[-1] - 1:-1])#这段切片操作提取了成熟层（mature_layer）在输入序列 input_ids 中，希望生成安全回复的 logits
            safe_logits = torch.zeros_like(dict_outputs[toxic_layer][0, prefix_ids.shape[-1] - 1:-1])#这段切片操作提取了成熟层（mature_layer）在输入序列 input_ids 中，希望生成安全回复的 logits

            base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
            bad_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
            # for i, l in enumerate(premature_layers):
            #     base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
            for i, (safe_layer, premature_layer) in enumerate(zip(safe_layers, premature_layers)):
                base_logits[i] = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1 + i]
                safe_logits[i] = dict_outputs[safe_layer][0, prefix_ids.shape[-1] - 1 + i]#将每一句话的logit值赋给 base_logits[i]，其中 i 是当前层在 presafe_layers 列表中的索引。

                bad_logits[i] =  base_logits[i] - safe_logits[i]
            toxic_logits = dict_outputs[toxic_layer][0,prefix_ids.shape[-1] - 1:-1]
            final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]
            good_logits = final_logits
            # bad_logits = base_logits+toxic_logits-safe_logits
            final_logits = final_logits.log_softmax(dim=-1)
            
            # bad_logits = base_logits-safe_logits
            base_logits = base_logits.log_softmax(dim=-1)

            toxic_logits = toxic_logits.log_softmax(dim=-1)
            good_logits = good_logits.log_softmax(dim=-1)
            bad_logits = bad_logits.log_softmax(dim=-1)
            different_logits = good_logits - bad_logits

            safe_logits = safe_logits.log_softmax(dim=-1)

            diff_logits = final_logits-bad_logits

            if post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)
            # log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
                #这里体现的APC，实现了对预测概率的自适应过滤，提高了生成文本的事实性
            # print("relative_top："+str(relative_top))
            if relative_top > 0.0:

                relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)#
                # 获取前 10% 的索引
                diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)

                true_indices = torch.nonzero(relative_top_mask, as_tuple=False)

                # 输出前 10% 的索引
                # print("True indices (top 10%):", true_indices)

                # 验证选中的数量是否为 10%
                total_elements = final_logits.numel()
                selected_elements = true_indices.size(0)

                print("Percentage of selected elements:", selected_elements / total_elements * 100, "%")


            log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()#8.9日修改

            diff_logits = diff_logits.unsqueeze(0)


        return log_probs,(premature_layer_dist if mode == 'dscd' else None)             # print("dict_outputs:"+str(dict_outputs))
    def Logits_T(self, input_text1, input_text2,toxic_layer, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):

        with torch.no_grad():

            input_text = input_text1 + input_text2


            input_ids = self.tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(self.device)#上下文和希望生成的安全答案
            prefix_ids = self.tokenizer(input_text1, return_tensors="pt", padding=True).input_ids.to(self.device)#上下文,即prefix

            # input_ids = dict(self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device))#上下文和希望生成的安全答案
            input_text2_ids = self.tokenizer(input_text2, return_tensors="pt", padding=True).to(self.device).input_ids.to(self.device)#上下文和希望生成的安全答案
            # prefix_ids = self.tokenizer(input_text1, return_tensors="pt", padding=True).input_ids.to(self.device)#上下文,即prefix
            # input_ids = prefix_ids
            out_labels =input_text2_ids.masked_fill(input_text2_ids == self.tokenizer.pad_token_id, -1000)

            continue_ids = input_ids[0, prefix_ids.shape[-1]:]
            # continue_ids = input_ids[0,:]

                
            # premature_layer_dist = {l:0 for l in candidate_premature_layers}
            premature_layer_dist = {l:0 for l in candidate_premature_layers}


            picked_logits = []
            result_dict = {}
            premature_layers = []
            safe_layers = []

            count=0
            dict_outputs, outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                early_exit_layers=candidate_premature_layers+[toxic_layer]+[mature_layer],
                
            )
            safe_layers_dist =candidate_premature_layers+[mature_layer]
            safe_layer_dist = {l:0 for l in safe_layers_dist}


            # for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):#处理了希望生成的安全答案的token,比较了每个token其成熟层与其差异最大的过早层之间的logits
            # for seq_i in range(1, input_ids.shape[-1] - 1):#处理了希望生成的安全答案的token,比较了每个token其成熟层与其差异最大的过早层之间的logits
            for seq_i in range(prefix_ids.shape[-1] - 1, input_ids.shape[-1] - 1):

                # Pick the less like layer to contrast with
                # 1. Stacking all presafe_layers into a new dimension

                stacked_premature_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in candidate_premature_layers], dim=0)
                # print("candidate_presafe_layers:{}".format(candidate_premature_layers[1:]))
                stacked_presafe_layers = torch.stack([dict_outputs[i][:, seq_i, :] for i in safe_layers_dist], dim=0)#将候选过早层所有的隐藏状态叠加到第0维,根据bert rediscovers这篇论文的3.2节中提到，第0层信息太琐碎，要从第1层开始
                number_of_layers = len(candidate_premature_layers)  # 使用len()函数获取列表长度

                # print(number_of_layers) 
                num_iterations = input_ids.shape[-1] - 2  # 减去2是因为range的结束索引是倒数第二个
                num = number_of_layers*num_iterations


                # print("stacked_presafe_layers："+str(stacked_presafe_layers))#调试，过早层的选择！
                # 2. Calculate the softmax values for mature_layer and all presafe_layers
                # softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                ##########修改8.12,有毒层的加入
                softmax_toxic_layer = F.softmax(dict_outputs[toxic_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)

                softmax_mature_layer = F.softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)

                
                softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)  # shape: (num_presafe_layers, batch_size, num_features)

                softmax_presafe_layers = F.softmax(stacked_presafe_layers, dim=-1)  # shape: (num_presafe_layers, batch_size, num_features)

                # 3. Calculate M, the average distribution
                M = 0.5 * (softmax_toxic_layer[None, :, :]+ softmax_presafe_layers)  # shape: (num_presafe_layers, batch_size, num_features)
                N = 0.5 * (softmax_mature_layer[None, :, :] + softmax_premature_layers)  # shape: (num_premature_layers, batch_size, num_features)


                # 4. Calculate log-softmax for the KL divergence
                log_softmax_toxic_layer = F.log_softmax(dict_outputs[toxic_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                # log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                log_softmax_presafe_layers = F.log_softmax(stacked_presafe_layers, dim=-1)  # shape: (num_presafe_layers, batch_size, num_features)
#############################
                log_softmax_mature_layer = F.log_softmax(dict_outputs[mature_layer][:, seq_i, :], dim=-1)  # shape: (batch_size, num_features)
                log_softmax_premature_layers = F.log_softmax(stacked_premature_layers, dim=-1)  # shape: (num_premature_layers, batch_size, num_features)
                # 5. Calculate the KL divergences and then the JS divergences
                kl3 = F.kl_div(log_softmax_mature_layer[None, :, :], N, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                kl4 = F.kl_div(log_softmax_premature_layers, N, reduction='none').mean(-1)  # shape: (num_premature_layers, batch_size)
                js_divs_mature = 0.5 * (kl3 + kl4)  # shape: (num_premature_layers, batch_size)
##########################
                # 5. Calculate the KL divergences and then the JS divergences
                kl1 = F.kl_div(log_softmax_toxic_layer[None, :, :], M, reduction='none').mean(-1)  # shape: (num_presafe_layers, batch_size)，计算成熟层与M的kl散度
                kl2 = F.kl_div(log_softmax_presafe_layers, M, reduction='none').mean(-1)  # shape: (num_presafe_layers, batch_size)，计算过早层与M的kl散度，
                js_divs_safe = 0.5 * (kl1 + kl2)  # shape: (num_presafe_layers, batch_size)

                # 6. Reduce the batchmean
                js_divs_safe_mean  = js_divs_safe.mean(-1)  # shape: (num_presafe_layers,)
                # 将第 0 层的 JS 散度值设为负无穷，确保不会被选择
                js_divs_safe_mean[0] = float('-inf')

                # 计算 topk(2) 后，第 0 层不会出现在 top_two_indices 中
                top_two_indices = js_divs_safe_mean.topk(2).indices.cpu().tolist()

                # 选择 safe_layer，确保不会选择第 0 层
                safe_layer = candidate_premature_layers[top_two_indices[0]]
                # print("safe_layer:{}".format(safe_layer))
                # print("safe_layer:{}".format(safe_layer))
                safe_layer_dist[safe_layer] += 1
###############################################
                # 6. Reduce the batchmean
                js_divs_mature = js_divs_mature.mean(-1)  # shape: (num_premature_layers,)
                premature_layer = candidate_premature_layers[int(js_divs_mature.argmax().cpu().item())]
                premature_layer_dist[premature_layer] += 1

                premature_layers.append(premature_layer)
###########################################
                safe_layers.append(safe_layer)
                
                # safe_logits+= dict_outputs[safe_layer][0, prefix_ids.shape[-1] - 1:-1]
                # print("safe_layer:{},safe_logits: {}".format(safe_layer,dict_outputs[safe_layer][0, prefix_ids.shape[-1] - 1:-1]))
                # count+=1

            # counter_safe  = Counter(safe_layers)
            # counter_premature  = Counter(premature_layers)
            # 计算每个元素的出现次数
            # counter = Counter(data)

            # 找出出现次数最多的元素
            # SAFE, SAFE_count = counter_safe.most_common(1)[0]
            # premature, premature_count = counter_premature.most_common(1)[0]

            # print(f"safe出现次数最多的元素是: {SAFE}")

            # safe_logits = safe_logits/count

            safe_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])

            base_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])
            for i, l in enumerate(premature_layers):
                base_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
            for i, l in enumerate(safe_layers):
                safe_logits[i] = dict_outputs[l][0, prefix_ids.shape[-1] - 1 + i]
            toxic_logits = torch.zeros_like(dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1])

            toxic_logits = dict_outputs[toxic_layer][0,prefix_ids.shape[-1] - 1:-1]
            # safe_logits = dict_outputs[SAFE][0,prefix_ids.shape[-1] - 1:-1]
            # base_logits = dict_outputs[premature][0,prefix_ids.shape[-1] - 1:-1]
            final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1:-1]

            good_logits = final_logits+safe_logits
            bad_logits = base_logits-safe_logits
            # final_logits = final_logits.log_softmax(dim=-1)
            # base_logits = base_logits.log_softmax(dim=-1)
            safe_logits = safe_logits.log_softmax(dim=-1)

            toxic_logits = toxic_logits.log_softmax(dim=-1)

            good_logits = good_logits.log_softmax(dim=-1)
            bad_logits = bad_logits.log_softmax(dim=-1)
            # different_logits = good_logits - bad_logits

            final_logits = final_logits.log_softmax(dim=-1)
            toxic_logits = toxic_logits.log_softmax(dim=-1)
            diff_logits = final_logits-bad_logits
            if post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)
            if relative_top > 0.0:

                relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)#

                diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)



                true_indices = torch.nonzero(relative_top_mask, as_tuple=False)

                # 验证选中的数量是否为 10%
                total_elements = final_logits.numel()
                selected_elements = true_indices.size(0)

                print("Percentage of selected elements:", selected_elements / total_elements * 100, "%")

            log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()#8.9日修改

            diff_logits = diff_logits.unsqueeze(0)

        return log_probs,(premature_layer_dist if mode == 'dscd' else None)             # print("dict_outputs:"+str(dict_outputs))