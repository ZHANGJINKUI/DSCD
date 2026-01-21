from typing import Optional, Union, List, Tuple, Dict
from time import time
from tqdm import tqdm
import json
import torch
import numpy as np
import random
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from ..evaluate import compute_safety_edit_quality, ccks_compute_safety_edit_quality
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *

# hparams_path="/mnt/sdb/zjk/DOLA_onlyone/hparams/DINM/llama-7b.yaml"#注释：祛毒编辑的模型参数设置

# hparams = DINMHyperParams.from_hparams(hparams_path)
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)

LOG = logging.getLogger(__name__)

def make_logs():

    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)

def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
seed_everything(42)
# def _locate_toxic_layer(model, tokenizer, requests, hparams, **kwargs):
#     # if isinstance(tokenizer, LlamaTokenizer):
#     #     tokenizer.padding_side = 'right'
#     # else:
#     #     tokenizer.padding_side = 'left'
#     toxic_layer = []
#     # input = tokenizer([value for pair in requests for value in [pair["target_new"], pair["ground_truth"]]], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{self.hparams.device}") 
#     input = tokenizer(requests, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{hparams.device}") 

#     with torch.no_grad():
#         outputs = model(**input)
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
    # if isinstance(tokenizer, LlamaTokenizer):
    #     tokenizer.padding_side = 'right'
    # else:
    #     tokenizer.padding_side = 'left'
    toxic_layers = []
    # input = tokenizer([value for pair in requests for value in [pair["target_new"], pair["ground_truth"]]], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{self.hparams.device}") 
    input = tokenizer(requests, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{hparams.device}") 

    with torch.no_grad():
            outputs = model(**input)
    hidden_states = outputs.hidden_states
    print(len(outputs.hidden_states))
    for j in range(1):
        max_distances = [float('-inf'), float('-inf')]  # 用于存储最大和次大距离
        max_indices = [None, None]  # 用于存储对应的层索引

        for layer_index in range(1, len(hidden_states)):
            euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[layer_index][j * 2 + 1], p=2)

            if euclidean_distance.item() > max_distances[0]:  # 如果是最大距离
                max_distances[1] = max_distances[0]
                max_indices[1] = max_indices[0]
                max_distances[0] = euclidean_distance.item()
                max_indices[0] = layer_index
            elif euclidean_distance.item() > max_distances[1]:  # 如果是次大距离
                max_distances[1] = euclidean_distance.item()
                max_indices[1] = layer_index
        if((max_indices[0] - 1)%2==0):
            toxic_layers.append(max_indices[0] - 1)
        else:
            toxic_layers.append(max_indices[1] - 1)
            
    # 保存到JSON文件
    save_toxic_layer_to_json(max_indices[0] - 1, "/home/zjk/Program/EasyEdit_dola/Mistral_safe_new/toxic_layers.json")
    save_toxic_layer_to_json(max_indices[1] - 1, "/home/zjk/Program/EasyEdit_dola/Mistral_safe_new/toxic_candidate.json")

    return toxic_layers  # 返回形式为 [4, 5]

def save_toxic_layer_to_json(toxic_layer_value, json_file):
    # 检查文件是否存在
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
    else:
        data = []

    # 追加新的toxic_layer值
    data.append(toxic_layer_value)

    # 写回JSON文件
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)
# class SafetyEditor(BaseEditor)
class SafetyEditor:

    @classmethod
    def from_hparams(cls, hparams: HyperParams):

        return cls(hparams)

    def __init__(self,
                hparams: HyperParams,
                 ):

        assert hparams is not None, print('Error: hparams is None.')

        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name

        make_logs()

        LOG.info("Instantiating model")

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = torch.bfloat16
            # torch_dtype = torch.float16 if hasattr(hparams, 'fp16') and hparams.fp16 else torch.float32
            print(torch_dtype)
            if 'llama' in self.model_name.lower():
                # local_model_path = '/home/zjk/llama2/EasyEdit/llama2-7b-chat'
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map='auto',max_memory= {i: f"{22}GiB" for i in range(1)})
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.model_max_length = 1024 # 或根据你的模型设置一个合理值

                # self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True,  torch_dtype=torch_dtype, device_map='auto')
                # self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
                print(self.tok.model_max_length)

            elif 'mistral' in self.model_name.lower():
                # self.model = MistralForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map='auto',max_memory= {i: f"{23}GiB" for i in range(1)})
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map='auto',max_memory= {i: f"{22}GiB" for i in range(1)})
                self.tok = AutoTokenizer.from_pretrained(self.model_name)

                # self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id 
            elif 'qwen' in self.model_name.lower():
                # self.model = MistralForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map='auto',max_memory= {i: f"{23}GiB" for i in range(1)})
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map='auto',max_memory= {i: f"{22}GiB" for i in range(1)})
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.model_max_length = 1024 # 或根据你的模型设置一个合理值

                # self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True,  torch_dtype=torch_dtype, device_map='auto')
                # self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'falcon' in self.model_name.lower():
                # self.model = MistralForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, low_cpu_mem_usage=True, torch_dtype=torch_dtype, device_map='auto',max_memory= {i: f"{23}GiB" for i in range(1)})
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map='auto',max_memory= {i: f"{22}GiB" for i in range(1)})
                self.tok = AutoTokenizer.from_pretrained(self.model_name)

                # self.tok = LlamaTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id 
            elif 'gpt' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, output_hidden_states=True, torch_dtype=torch_dtype, device_map=device_map,max_memory= {i: f"{22}GiB" for i in range(1)})
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id    
            else:
                raise NotImplementedError
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            hparams.device = str(self.model.device).split(":")[1]
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f"cuda:{hparams.device}") 

        self.hparams = hparams

    # def _locate_toxic_layer(self, model, tokenizer, requests, **kwargs):
    #     # if isinstance(tokenizer, LlamaTokenizer):
    #     #     tokenizer.padding_side = 'right'
    #     # else:
    #     #     tokenizer.padding_side = 'left'
    #     toxic_layers = []
    #     input = tokenizer([value for pair in requests for value in [pair["target_new"], pair["ground_truth"]]], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{self.hparams.device}") 
    #     with torch.no_grad():
    #         outputs = model(**input)
    #     hidden_states = outputs.hidden_states

    #     # 打印隐藏状态的数量，即层的数量
    #     print(len(hidden_states))
    #     for j in range(len(requests)):
    #         max_distances = [float('-inf'), float('-inf')]  # 用于存储最大和次大距离
    #         max_indices = [None, None]  # 用于存储对应的层索引

    #         for layer_index in range(1, len(hidden_states)):
    #             euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[layer_index][j * 2 + 1], p=2)

    #             if euclidean_distance.item() > max_distances[0]:  # 如果是最大距离
    #                 max_distances[1] = max_distances[0]
    #                 max_indices[1] = max_indices[0]
    #                 max_distances[0] = euclidean_distance.item()
    #                 max_indices[0] = layer_index
    #             elif euclidean_distance.item() > max_distances[1]:  # 如果是次大距离
    #                 max_distances[1] = euclidean_distance.item()
    #                 max_indices[1] = layer_index

    #         toxic_layers.extend([max_indices[0] - 1, max_indices[1] - 1])  # 扁平化添加最大和次大层索引
        
    #     return toxic_layers  # 返回形式为 [4, 5]
    def _locate_toxic_layer(self, model, tokenizer, requests, **kwargs):
        # if isinstance(tokenizer, LlamaTokenizer):
        #     tokenizer.padding_side = 'right'
        # else:
        #     tokenizer.padding_side = 'left'
        toxic_layers = []
        input = tokenizer([value for pair in requests for value in [pair["target_new"], pair["ground_truth"]]], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{self.hparams.device}") 
        with torch.no_grad():
            outputs = model(**input)
        hidden_states = outputs.hidden_states

        # 打印隐藏状态的数量，即层的数量
        print(len(hidden_states))
        for j in range(len(requests)):
            max_distances = float('-inf')
            max_indices = None  # 用于存储对应的层索引

            for layer_index in range(1, len(hidden_states)):
                euclidean_distance = torch.dist(hidden_states[layer_index][j * 2], hidden_states[layer_index][j * 2 + 1], p=2)

                if euclidean_distance.item() > max_distances:  # 如果是最大距离
                    max_distances= euclidean_distance.item()
                    max_indices= layer_index

            toxic_layers.append(max_indices - 1)
        return toxic_layers

    def edit(self,
             save_path,
             prompts: Union[str, List[str]],
             prompts_with_systemPrompt: Union[str, List[str]],
             target_new: Union[str, List[str]],
             generate_kwargs,
             ground_truth: Optional[Union[str, List[str]]] = None,
             locality_inputs:  Optional[Dict] = None,
             locality_inputs_with_systemPrompt:  Optional[Dict] = None,
             general_prompt: Optional[Union[str, List[str]]] = None,
             general_prompt_with_systemPrompt: Optional[Union[str, List[str]]] = None,
             
             keep_original_weight=False,
             verbose=True,

             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for general knowledge constrains
        """
        # print("kwargs"+str(kwargs))

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts,], [target_new,]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth,]
            else:
                assert len(ground_truth) == len(prompts)
        else: # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = self._prepare_requests(prompts, target_new, ground_truth, general_prompt, locality_inputs, **kwargs)
            requests_with_systemPrompt = self._prepare_requests(prompts_with_systemPrompt, target_new, ground_truth, general_prompt_with_systemPrompt, locality_inputs_with_systemPrompt, **kwargs)

        if hasattr(self.hparams, 'batch_size') :
               assert self.hparams.batch_size == 1, print(f'Single Edit, pls set the batch_size to 1....')
        # request = target_new+ ground_truth
        # start_locate = time()
        # toxic_layers = _locate_toxic_layer(self.model, self.tok, request,self.hparams)#定位到有毒层，编辑的也是有毒层，输出显示的时候却+1，是为了让人知道是第几层吗？
        # exec_time_locate = time() - start_locate
        # # LOG.info(f"Execution Locate editing took {exec_time_locate}")
        # self.hparams.layers = [toxic_layers[0]]
        if "NLPCC" in kwargs and kwargs['NLPCC']:
            for i, (request, request_with_systemPrompt) in enumerate(zip(requests, requests_with_systemPrompt)):
                start = time()
                # if len(self.hparams.layers) == 0:
                # self.hparams.layers = self._locate_toxic_layer(self.model, self.tok, [request,],self.hparams)#定位到有毒层，编辑的也是有毒层，输出显示的时候却+1，是为了让人知道是第几层吗？
                # exec_locate_time = time()-start
                # edited_model, weights_copy = self.apply_algo(
                #     self.model,
                #     self.tok,
                #     [request_with_systemPrompt],
                #     self.hparams,
                #     copy=False,
                #     return_orig_weights=True,
                #     keep_original_weight=keep_original_weight,
                #     train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                # )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
                # edited_model.save_pretrained(kwargs['ckpt_save_dir'])
                # print(f"edited model is saved in {kwargs['ckpt_save_dir']}")
                # with torch.no_grad():
                #     for k, v in weights_copy.items():
                #         nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
              

        else:
            all_metrics = []
            run_times_Vanilla = []
            run_times_DSCD = []
            run_times_DINM = []
            run_times_DSCD_DINM = []

            if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
                metrics = kwargs['pre_edit']
                all_metrics = metrics
            else:
                for i, (request, request_with_systemPrompt) in enumerate(zip(requests, requests_with_systemPrompt)):
                    self.hparams.layers = [28]
                    # self.hparams.layers = self._locate_toxic_layer(self.model, self.tok, [request,])

                    # Phase 1: Evaluate Vanilla (vanilla - original model with standard decoding)
                    start_Vanilla = time()
                    if "ccks" in kwargs and kwargs['ccks']:
                        metrics = {
                            "Vanilla": ccks_compute_safety_edit_quality(self.model, self.tok, request_with_systemPrompt,
                                                    self.hparams.device,toxic_layer=self.hparams.layers,generate_kwargs=generate_kwargs, max_tokens=self.hparams.max_length, max_output_tokens=self.hparams.max_output_length)
                        }
                    else:
                        dscd = False
                        metrics = {
                            "Vanilla": compute_safety_edit_quality(dscd,save_path,self.model, self.tok, request_with_systemPrompt,
                                                    self.hparams.device,toxic_layer=self.hparams.layers,generate_kwargs=generate_kwargs, max_tokens=self.hparams.max_length, max_output_tokens=self.hparams.max_output_length)
                        }
                    exec_time_Vanilla = time() - start_Vanilla
                    LOG.info(f"Execution {i} Vanilla (vanilla) took {exec_time_Vanilla}")
                    run_times_Vanilla.append(exec_time_Vanilla)
                    
                    # Phase 2: Evaluate DSCD (original model with DSCD decoding)
                    start_DSCD = time()
                    if "ccks" in kwargs and kwargs['ccks']:
                        metrics.update({
                            "DSCD": ccks_compute_safety_edit_quality(self.model, self.tok, request_with_systemPrompt,
                                                    self.hparams.device,toxic_layer=self.hparams.layers,generate_kwargs=generate_kwargs, max_tokens=self.hparams.max_length, max_output_tokens=self.hparams.max_output_length)
                        })
                    else:
                        dscd = True
                        metrics.update({
                            "DSCD": compute_safety_edit_quality(dscd,save_path,self.model, self.tok, request_with_systemPrompt,
                                                    self.hparams.device,toxic_layer=self.hparams.layers,generate_kwargs=generate_kwargs, max_tokens=self.hparams.max_length, max_output_tokens=self.hparams.max_output_length)
                        })
                    exec_time_DSCD = time() - start_DSCD
                    LOG.info(f"Execution {i} DSCD took {exec_time_DSCD}")
                    run_times_DSCD.append(exec_time_DSCD)

                    all_metrics.append(metrics)

                    # file_name = f'/mnt/sdb/zjk/EasyEdit_dola/DSCDvsDINM_2/time_DSCD.json'

                    # try:
                    #     with open(file_name, 'r') as file:
                    #         run_times_DSCD = json.load(file)
                    # except FileNotFoundError:
                    #     run_times_DSCD = []
                    # # 将更新后的列表写回同一个JSON文件
                    # with open(file_name, 'w') as file:
                    #     json.dump(run_times_DSCD, file, indent=4)

                    # print(f"已将新时间添加到列表并保存")

                    # # 将列表转换为JSON格式并写入文件
                    # with open(file_name, 'w') as file:
                    #     json.dump(run_times_DSCD, file)

                    # print(f"已将新时间 {exec_time_DSCD} 添加到列表并保存")
                if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                    ### Store the pre_edit metric to refrain computing repeatedly
                    json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)
                # with torch.no_grad():
                #     for k, v in weights_copy.items():
                #         nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
            for i, (request, request_with_systemPrompt) in enumerate(zip(requests, requests_with_systemPrompt)):
                start_DINM = time()
                # if len(self.hparams.layers) == 0:
                self.hparams.layers = [28]
                # self.hparams.layers = self._locate_toxic_layer(self.model, self.tok, [request,])

                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request_with_systemPrompt],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                exec_time = time() - start_DINM
                # LOG.info(f"Execution update editing took {exec_time}")

                # start = time()
                if "ccks" in kwargs and kwargs['ccks']:
                    all_metrics[i].update({
                        'case_id': kwargs["case_id"],
                        "requested_rewrite": request,
                        "DINM": ccks_compute_safety_edit_quality(self.model, self.tok, request_with_systemPrompt, self.hparams.device, toxic_layer=self.hparams.layers,generate_kwargs=generate_kwargs,max_tokens=self.hparams.max_length, max_output_tokens=self.hparams.max_output_length),
                        "time": exec_time,
                    })

                else:
                    dscd = True
                    start_DINM = time()
                    all_metrics[i].update({
                        'case_id': kwargs["case_id"],
                        "requested_rewrite": request,
                        "DINM": compute_safety_edit_quality(dscd,save_path,edited_model, self.tok, request_with_systemPrompt, self.hparams.device, toxic_layer=self.hparams.layers,generate_kwargs=generate_kwargs,max_tokens=self.hparams.max_length, max_output_tokens=self.hparams.max_output_length),
                        "time": exec_time,
                    })
                exec_time_DINM = time() - start_DINM
                LOG.info(f"Execution ORI editing took {exec_time_DINM+exec_time}")
                run_times_DINM.append(exec_time_DINM+exec_time)
                
                # Phase 4: Evaluate DSCD+DINM (edited model with DSCD decoding)
                start_DSCD_DINM = time()
                dscd = False
                all_metrics[i].update({
                    "DSCD+DINM": compute_safety_edit_quality(dscd,save_path,edited_model, self.tok, request_with_systemPrompt, self.hparams.device, toxic_layer=self.hparams.layers,generate_kwargs=generate_kwargs,max_tokens=self.hparams.max_length, max_output_tokens=self.hparams.max_output_length)
                })
                exec_time_DSCD_DINM = time() - start_DSCD_DINM
                LOG.info(f"Execution DSCD+DINM took {exec_time_DSCD_DINM}")
                run_times_DSCD_DINM.append(exec_time_DSCD_DINM)
                
                # Add case_id and requested_rewrite to Vanilla and DSCD metrics
                all_metrics[i]['case_id'] = kwargs["case_id"]
                all_metrics[i]['requested_rewrite'] = request
                # file_name = f'/mnt/sdb/zjk/EasyEdit_dola/DSCDvsDINM_2/time_DINM.json'
                # try:
                #     with open(file_name, 'r') as file:
                #         run_times_DINM = json.load(file)
                # except FileNotFoundError:
                #     run_times_DINM = []
                # # 将列表转换为JSON格式并写入文件
                # with open(file_name, 'w') as file:
                #     json.dump(run_times_DINM, file)

                # print(f"已将新时间 {exec_time_DINM+exec_time_locate} 添加到列表并保存")
                # 恢复模型参数
                with torch.no_grad():
                    for k, v in weights_copy.items():
                        nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")
                

                    # LOG.info(f"Evaluation took {time() - start}")

                    # if verbose:
                    #     LOG.info(
                    #         f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    #     )

                # if isinstance(edited_model, LORA):
                #     edited_model=edited_model.model
                #for melo
                # print(all_metrics)
                return all_metrics, self.model,target_new,run_times_Vanilla,run_times_DSCD,run_times_DINM,run_times_DSCD_DINM

    def _prepare_requests(self,
                          prompts: Union[str, List[str]],
                          target_new: Union[str, List[str]],
                          ground_truth: Union[str, List[str]],
                          general_prompt: Optional[Union[str, List[str]]] = None,
                          locality_inputs: Optional[Dict] = None,
                          **kwargs
                          ):
        if general_prompt is None:
            requests = [{
                'prompt': prompt,
                'target_new': target_new_,
                'ground_truth': ground_truth_,
                'locality': {}
            }
            for prompt, ground_truth_, target_new_ in zip(prompts, ground_truth, target_new)
            ]
        
        else:

            requests = [{
                'prompt': prompt,
                'target_new': target_new_,
                'ground_truth': ground_truth_,
                'general_prompt': general_prompt_,
                'locality': {}
            }
            for prompt, ground_truth_, target_new_, general_prompt_ in zip(prompts, ground_truth, target_new, general_prompt)
            ]

        
        if locality_inputs is not None:
            for locality_key in locality_inputs.keys():
                if isinstance(locality_inputs[locality_key]['prompt'], str):
                    locality_inputs[locality_key]['prompt'] = [locality_inputs[locality_key]['prompt'],]
                    locality_inputs[locality_key]['ground_truth'] = [locality_inputs[locality_key]['ground_truth'], ]
                assert len(locality_inputs[locality_key]['prompt']) == len(locality_inputs[locality_key]['ground_truth']) \
                == len(requests), print('One Edit instance needs one locality input.....')

                for i, request in enumerate(requests):
                    if locality_inputs[locality_key]['prompt'][i] is not None:
                        request['locality'].update(
                            {
                                locality_key: {
                                    f'prompt': locality_inputs[locality_key]['prompt'][i],
                                    f'ground_truth': locality_inputs[locality_key]['ground_truth'][i]
                                }
                            }
                        )

        
        return requests
