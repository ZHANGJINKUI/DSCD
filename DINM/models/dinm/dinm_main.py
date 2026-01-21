from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
from ...trainer import kl_loc_loss, masked_log_probs
import torch
import torch.nn.functional as F
# import nethook
from . import nethook
import torch
import torch.nn.functional as F
import torch.nn as nn
from .dinm_hparams import DINMHyperParams
# from ...trainer import kl_loc_loss, masked_log_probs

#进行祛毒编辑修改
#    edit_model,_ = apply_dinm_to_model(model, tok, request, hparams,diff_logits,final_logits,toxic_logits,base_logits)

def apply_dinm_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DINMHyperParams,
    log_probs:float,
    diff_logits: torch.Tensor,  # 新增的 diff_logits 参数
    num:int,
      # 新增的参数1：层名称列表
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,


    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
# def apply_dinm_to_model(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     requests: List[Dict],
#     hparams: DINMHyperParams,
#     log_probs:torch.tensor,
#     copy=False,
#     return_orig_weights=False,
#     keep_original_weight=False,
#     **kwargs: Any,
# ) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 确保模型在 GPU 上
        # model = model.to(device)#8.10修改
    # deltas = execute_dinm(model, tok, requests, hparams,log_probs,diff_logits,final_logits,toxic_logits,base_logits)#将DOLA的损失函数加入DINM中，8.9修改
    deltas = execute_dinm(model, tok, requests, hparams,log_probs,diff_logits,num)#将DOLA的损失函数加入DINM中

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}
    print(f"Model is on {next(model.parameters()).device}")#8.10修改

    return model, weights_copy
def apply_dinm_to_model_ori(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DINMHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_dinm_ori(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy
#################################################
#修改8.8
# def apply_dinm_to_model(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     requests: List[Dict],
#     hparams: DINMHyperParams,
#     copy=False,
#     return_orig_weights=False,
#     keep_original_weight=False,


#     **kwargs: Any,
# ) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
#     """
#     Returns a model with the desired changes.
#     :param copy: If true, will preserve the original model while creating a new one to edit.
#         Note that you are responsible for deallocating the new model's memory to avoid leaks.
#     :return: (1) the updated model, (2) the weights that changed
#     """
#     weights_copy = {}
#     if copy:
#         model = deepcopy(model)
#     print(requests)
#     deltas = execute_dinm(model, tok, requests, hparams)

#     with torch.no_grad():
#         for w_name, upd_matrix in deltas.items():
#             w = nethook.get_parameter(model, w_name)
#             if return_orig_weights and w_name not in weights_copy:
#                 weights_copy[w_name] = w.detach().clone()

#             w[...] += upd_matrix

#     print(f"New weights successfully inserted into {list(deltas.keys())}")

#     if not keep_original_weight:
#         weights_copy = {}

#     return model, weights_copy

def get_edit_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -1000)




#返回的 deltas 是一个字典，它包含了模型权重的变化量
#deltas 表示在DINM过程中需要应用于模型参数的实际更改。正值表示参数需要增加，负值表示参数需要减少。这些更改旨在调整模型的行为，以抑制生成有害输出的能力。
#deltas 是DINM算法中计算出的权重更新量，它们表示了为了达到减少模型生成有害内容目的所需进行的权重调整。这些更新量可以被用来更新模型的权重，或者用于进一步分析DINM算法的效果。

def execute_dinm_ori(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DINMHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers   # specific layer for each instance
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights


    ######## general knowledge constraint#####################
    instruction_TextsandTargets = [r["locality"]["general knowledge constraint"]["prompt"] + " " + r["locality"]["general knowledge constraint"]["ground_truth"] for r in requests]
    with torch.no_grad():
            instructandAns = dict(
                tok(
                    instruction_TextsandTargets,
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)   #  torch.Size([1, 148])
            )
            instructonlyAns = dict(
                tok(
                    [r["locality"]["general knowledge constraint"]["ground_truth"] for r in requests],
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)  
            )  #  torch.Size([1, 59])
    instruction_base_Logits = model(**instructandAns).logits  # (B, L, D) (1,148,32000)
    instruction_base_Logits = instruction_base_Logits[:, -instructonlyAns["attention_mask"].size(1):]  #torch.Size([1, 59, 32000])
    
    ############edit toxic regions#############################
    # # Update loop: intervene at layers simultaneously
    # loss_meter = AverageMeter()
    ft_input = [request["prompt"] + " " + request["target_new"] for request in requests]
    out_ids = dict(tok(request["target_new"], return_tensors="pt", padding=True).to(device))  #torch.Size([1, 69]),希望生成的安全答案的token


    out_labels = get_edit_labels(tok, out_ids["input_ids"])

    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        inputs = tok(ft_input, return_tensors="pt", padding=True).to(device)
        opt.zero_grad()
        output = model(**inputs).logits  #torch.Size([1, 321, 32000])
        loss_dict = masked_log_probs(hparams, output, out_labels, shift=True)
        l_edit = loss_dict["nll"]
        with torch.no_grad():
            post_logits = model(**instructandAns).logits  # (B, L, D) tensor (1,59,32000)
        kl_mask = instructonlyAns["attention_mask"]
        if kl_mask.size(1) != post_logits.size(1):  #torch.Size([1, 59, 32000])
            post_logits = post_logits[:, -kl_mask.size(1):]   #torch.Size([1, 59, 32000])
        l_loc_instruction = kl_loc_loss(instruction_base_Logits.detach(), post_logits, mask=kl_mask) # tensor 一个值 0
        loss = hparams.kl_factor  * l_edit + l_loc_instruction
        # loss =  l_edit 
        print(f"Batch loss {loss.item()}, loss_edit*0.1:{0.1 * l_edit}, loss_loc_instruction:{l_loc_instruction}")

        if loss.item() >= 1e-4:
            loss.backward()
            opt.step()
            

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )
        else:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def execute_dinm(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DINMHyperParams,
    log_probs:float,
    diff_logits: torch.Tensor,  # 新增的 diff_logits 参数
    num:int,
      # 新增的参数1：层名称列表
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
# def execute_dinm(
#     model: AutoModelForCausalLM,
#     tok: AutoTokenizer,
#     requests: List[Dict],
#     hparams: DINMHyperParams,
#     log_probs:torch.tensor,
#     **kwargs: Any,
# ) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    
    # Retrieve weights that user desires to change检索用户想要更改的权重
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers   # specific layer for each instance
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights


    ######## general knowledge constraint#####################
    #知识编辑了
    instruction_TextsandTargets = [r["locality"]["general knowledge constraint"]["prompt"] + " " + r["locality"]["general knowledge constraint"]["ground_truth"] for r in requests]
    with torch.no_grad():
            instructandAns = dict(
                tok(
                    instruction_TextsandTargets,
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)   #  torch.Size([1, 148])
            )
            instructonlyAns = dict(
                tok(
                    [r["locality"]["general knowledge constraint"]["ground_truth"] for r in requests],
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)  
            )  #  torch.Size([1, 59])
    # instructandAns = {k: v.to('cuda:0') for k, v in instructandAns.items() if k != 'token_type_ids'}

#     instruction_base_Logits = model(
#     input_ids=instructandAns['input_ids'],
#     attention_mask=instructandAns['attention_mask']
# ).logits
    instruction_base_Logits = model(**instructandAns).logits  # (B, L, D) (1,148,32000)#8.9修改   

    instruction_base_Logits = instruction_base_Logits[:, -instructonlyAns["attention_mask"].size(1):]  #torch.Size([1, 59, 32000])
    
    ############edit toxic regions#############################
    # # Update loop: intervene at layers simultaneously
    # loss_meter = AverageMeter()
    ft_input = [request["prompt"] + " " + request["target_new"] for request in requests]
    out_ids = tok(request["target_new"], return_tensors="pt", padding=True).input_ids.to(device)  #torch.Size([1, 69])
    print("requests[target_new]:{}, type: {}".format(request["target_new"], type(request["target_new"])))

    print("out_ids.shape: {}, type: {}".format(out_ids.shape, type(out_ids.shape)))

    print("request[target_new]:"+str(request["target_new"]))



    out_labels = get_edit_labels(tok, out_ids)
    print("out_labels.shape:"+str(out_labels.shape))




    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        inputs = tok(ft_input, return_tensors="pt", padding=True).to(device)
        # inputs = {k: v.to('cuda:0') for k, v in inputs.items() if k != 'token_type_ids'}
        # 打印每个张量的形状


        for key, value in inputs.items():
            print(f"{key}: {value.size()}")  # 例如 torch.Size([1, 321, 32000])

        opt.zero_grad()
        output = model(**inputs).logits  #torch.Size([1, 321, 32000])



#         output = model(#修改8.9
#     input_ids=output['input_ids'],
#     attention_mask=output['attention_mask']
# ).logits
#masked_log_probs：这个函数根据预测（pred）和目标（targ）的维度调用不同的日志概率计算函数
        loss_dict = masked_log_probs(hparams, output, out_labels, shift=True)
        l_edit = loss_dict["nll"]
        with torch.no_grad():
            post_logits = model(**instructandAns).logits  # (B, L, D) tensor (1,59,32000)
        kl_mask = instructonlyAns["attention_mask"]
        if kl_mask.size(1) != post_logits.size(1):  #torch.Size([1, 59, 32000])
            post_logits = post_logits[:, -kl_mask.size(1):]   #torch.Size([1, 59, 32000])
        l_loc_instruction = kl_loc_loss(instruction_base_Logits.detach(), post_logits, mask=kl_mask) # tensor 一个值 0
# ###################################################################DOLA+DINM,得到我的diff_logits来进行新损失函数的设计
# ###################################################################DOLA+DINM,得到我的diff_logits来进行新损失函数的设计
        print("diff_logits_before.shape:"+str(diff_logits.shape))
        # print("out_labels.shape:"+str(out_labels.shape))

        loss_diff = masked_log_probs(hparams, diff_logits, out_labels, shift=True)
        print("diff_logits_after.shape:"+str(diff_logits.shape))
        print("log_probs:"+str(log_probs))

        loss_diff_all = loss_diff["nll"]
        loss_diff = loss_diff_all/num

        # 打印有效的 diff_logits 的形状

    #####################################################DOLA+DINM
        # loss = hparams.kl_factor  * l_edit + l_loc_instruction
        loss = hparams.kl_factor  * (l_edit+loss_diff) + l_loc_instruction#8.11修改
        # loss = hparams.kl_factor  * l_edit + l_loc_instruction#8.11修改


        # loss =  l_edit 
        print(f"Batch loss {loss.item()}, loss_edit*0.1:{0.1 * l_edit}, loss_loc_instruction:{l_loc_instruction},loss_diff*0.1:{0.1 * loss_diff}")
        # print(f"Batch loss {loss.item()}, loss_edit*0.1:{0.1 * l_edit}, loss_loc_instruction:{l_loc_instruction}")


        if loss.item() >= 1e-4:
            loss.backward()
            opt.step()
            

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )
        else:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas



def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
