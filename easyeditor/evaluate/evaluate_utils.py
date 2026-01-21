import os
import json
import torch
import numpy as np
import scipy
import nltk
import typing
from ..util.generate import generate_fast
import torch.nn.functional as F
from ..trainer import *
from sklearn.metrics import f1_score
# import openai  # Not used in safety evaluation
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria
from typing import List, Tuple

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
def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        if locality:
            return ans

        return np.mean(np.equal(ans, target))

def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    # prompt_tok = tok(
    #     prompts,
    #     padding=True,
    #     truncation=True,
    #     max_length=hparams.max_length,
    #     return_tensors="pt",
    # ).to("cpu")
    trg_tok = tok(
        targets,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")
    # trg_tok = tok(
    #     targets,
    #     padding=True,
    #     truncation=True,
    #     max_length=hparams.max_length,
    #     return_tensors="pt",
    # ).to("cpu")
    prompt_tok['decoder_input_ids'] = trg_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            answers = ans.squeeze().detach().cpu().numpy().tolist()
            return answers if type(answers[0]) is list else [answers,]
        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()

def test_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False, vanilla_generation=False):
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        results = []
        for prompt, target_new in zip(prompts, targets):
            target_new_tokens = tok.encode(target_new, add_special_tokens=False)
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(device)
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens),
                pad_token_id=tok.eos_token_id,
                use_cache=False,
            )
            if locality:
                results.append(gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])
            else:
                results.append(np.mean(np.equal(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):])))
        return results

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)
        if locality:
            return answers if type(answers[0]) is list else [answers,]
        if isinstance(answers[0], list):
            res = []
            for ans,label in zip(answers,labels):
                temp_acc = np.mean(np.equal(ans, label))
                if np.isnan(temp_acc):
                    continue
                res.append(temp_acc)
            return res
        else:
            return [np.mean(np.equal(answers, labels))]

def test_generation_quality_serac(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,       
):
    #only single case
    prompt_tok = tok(
        prefixes,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    prompt_tok_length=len(prompt_tok['input_ids'])
    gen_texts=model.generate(**prompt_tok,max_new_tokens=256)
    if isinstance(model,SERAC):
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))
    else:
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))      
    ngram_entropy = n_gram_entropy(gen_texts, return_list=True)


    ret = {
        "ngram_entropy": ngram_entropy
    }
    return ret

def test_generation_quality(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,
    vanilla_generation: bool = False,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=max_out_len,
        vanilla_generation=vanilla_generation,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    ret = {
        "ngram_entropy": ngram_entropy,
    }
    return ret

def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()

def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)

def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)

def PPL(
    model,
    tok,
    prompt: typing.Union[str, typing.List[str]],
    target_new: typing.Union[str, typing.List[str]],
    device,
):
    if isinstance(prompt, str):
        prompt,target_new = [prompt,], [target_new,]
    full_prompt = [f"{p} {l}" for p, l in zip(prompt, target_new)]
    prompt_ids = tok(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    for i in range(len(prompt)):
        tokens["labels"][i][:num_prompt_toks[i]] = -100
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = -100 # What is this doing?
    batch = {f"{k1}" : v1 for k1, v1 in tokens.items()}
    input_ids = batch["input_ids"][:, :1024]#.to(device)
    if "labels" not in batch:
        target_ids = batch["input_ids"][:, :1024].clone()
    else:
        target_ids = batch["labels"][:, :1024].clone()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), labels=target_ids.to(device))
        nll = outputs.loss
    ppl = torch.exp(nll)#.clip(0, 100)
    return ppl.cpu().numpy().tolist()

def verify_answer(model_answer, correct_answer):
    if type(correct_answer) is str:
        correct_answer = [[correct_answer]]
    for answer in correct_answer:
        if True not in [possible_answer in model_answer for possible_answer in answer]:
            return False
    return True

def answer_match(
    model,
    tok,
    prompt: str,
    target_new: str,
    device,
):
    inputs = tok.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, temperature=0, max_new_tokens=30)
    predict = tok.decode(outputs[0], skip_special_tokens=True)

    return verify_answer(predict,target_new)

def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]

def gather_log_probs(logits, labels):
    # print(f"labels.shape: {labels.shape} , logits.shape[:-1] :{logits.shape[:-1]}")
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def es(pre_logits, edit_logits, q_mask, labels, same_mask):
    
    _, targ = mask_hf_labels(labels)

    pos_mask = same_mask.unsqueeze(-1) * q_mask 
    neg_mask = (~same_mask).unsqueeze(-1) * q_mask 
        
    pre_token_log_probs = gather_log_probs(pre_logits, targ)
    edit_token_log_probs = gather_log_probs(edit_logits, targ)

    mean_pos_pre = masked_mean(pre_token_log_probs, pos_mask)
    mean_pos_edit = masked_mean(edit_token_log_probs, pos_mask)
    mean_neg_edit = masked_mean(edit_token_log_probs, neg_mask)

    z_sent = (mean_pos_edit - mean_neg_edit).sigmoid()
    z_topic_raw = (mean_pos_edit - mean_pos_pre).exp()
    z_topic = min(1, z_topic_raw)

    es_sent = z_sent * z_topic
    return es_sent

def es_per_icl(example, pre_logits, edit_logits):
    with torch.no_grad():
        
        pre_q_mask = example["outer_pre"]["q_mask"]
        edit_q_mask = example["outer_edit"]["q_mask"]
        
        pre_labels = example["outer_pre"]["labels"]
        edit_labels = example["outer_edit"]["labels"]
        
        pre_mask, pre_targ = mask_hf_labels(pre_labels)
        edit_mask, edit_targ = mask_hf_labels(edit_labels)
        
        same_per_mask = example["same_per_mask"]

        pre_pos_mask = same_per_mask.unsqueeze(-1) * pre_q_mask 
        pre_neg_mask = (~same_per_mask).unsqueeze(-1) * pre_q_mask 
        edit_pos_mask = same_per_mask.unsqueeze(-1) * edit_q_mask 
        edit_neg_mask = (~same_per_mask).unsqueeze(-1) * edit_q_mask 
        
        pre_token_log_probs = gather_log_probs(pre_logits, pre_targ)
        edit_token_log_probs = gather_log_probs(edit_logits, edit_targ)

        mean_pos_pre = masked_mean(pre_token_log_probs, pre_pos_mask)
        mean_pos_edit = masked_mean(edit_token_log_probs, edit_pos_mask)
        mean_neg_edit = masked_mean(edit_token_log_probs, edit_neg_mask)

        z_per = (mean_pos_edit - mean_neg_edit).sigmoid()
        z_topic_raw = (mean_pos_edit - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_per = z_per * z_topic
        return {
            "acc_per": es_per,
            "z_per": z_per,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_edit,
            "wrong_probs": mean_neg_edit,
        }

def per_generation(
    model,
    tok,
    max_out_len: int,
    target_per, 
    device,
    edited_model=None,
    IKE=False,
    **kwargs
    ):
    def generate_text(query, model, tokenizer):
        input_text = query
        generation_config = {
            "max_new_tokens": max_out_len,
            "temperature": 0,
            "eos_token_id": tokenizer.eos_token_id,
        }
        src_input_ids = tokenizer(input_text).input_ids
        input_ids = torch.tensor([src_input_ids], dtype=torch.long, device=device)
        outputs = model.generate(input_ids, **generation_config)
        response = tokenizer.decode(outputs[0][len(src_input_ids) :], skip_special_tokens=True)
        return response
    
    def clean_text(text):
        return text.strip().split("\n")[0]
    
    if IKE:
        pre_text = clean_text(generate_text(kwargs["pre_q"], model, tok))
        edit_text = clean_text(generate_text(kwargs["edit_q"], model, tok))

    else:
        assert edited_model is not None
        pre_text = clean_text(generate_text(kwargs["inner_q"], model, tok))
        edit_text = clean_text(generate_text(kwargs["inner_q"], edited_model.model, tok))

    ngram_pre_text = n_gram_entropy([pre_text])
    ngram_edit_text = n_gram_entropy([edit_text])
    coherent = ngram_pre_text >= 3.5 and ngram_edit_text >= 3.5
    
    result = {
        "pre_text": pre_text,
        "edit_text": edit_text,
        "ngram_pre_text": ngram_pre_text,
        "ngram_edit_text": ngram_edit_text,
        "coherent": coherent,
        "target_per": target_per,
    }

    return result

def kl_loc_loss(pre, post, mask=None):
    
    pre = pre.to(torch.float32).contiguous()
    post = post[:,-pre.shape[1]:,:].to(torch.float32).contiguous()
    
    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        # print("sequence")
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError

def F1(model, tok, hparams, prompts, targets, device, locality=False, vanilla_generation=True):
    if vanilla_generation:
        target_new_tokens = tok.encode(targets, add_special_tokens=False)
        prompt_tok = tok(
            prompts,
            return_tensors="pt",
        ).to(device)
        gen_token = model.generate(
            input_ids=prompt_tok['input_ids'],
            attention_mask=prompt_tok['attention_mask'],
            max_new_tokens=len(target_new_tokens),
            pad_token_id=tok.eos_token_id,
            use_cache=False,

        )
        return f1_score(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):], average='macro')
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)

        return f1_score(answers, labels, average='macro')

def test_instance_change(model, tok, max_length, prompts, targets, device, P = None):
    demo1_str = "Whether FrancoAngeli belongs to category publisher? Yes\nWhether And Other Stories belongs to category people? No\n"
    if P is None:
        prompts = demo1_str +prompts
    else:
        prompts = P + demo1_str + prompts

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    with torch.no_grad():
        pre_edit_outputs = model.generate(
            input_ids=prompt_tok['input_ids'].to(f"cuda:{device}"),
            attention_mask=prompt_tok['attention_mask'].to(f"cuda:{device}"),            
            # input_ids=prompt_tok['input_ids'].to("cpu"),
            # attention_mask=prompt_tok['attention_mask'].to("cpu"),
            max_new_tokens=2,
            pad_token_id=tok.eos_token_id
        )

        model_response = [tok.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        answer = model_response[0][model_response[0].rfind('?')+2:]
        # print(model_response[0], answer)

        if "yes" in answer.lower():
            return np.ones(1)
        else:
            if "no" not in answer.lower():
                print(f"entity error in define yes or no: {answer}")
                return np.array([-1.0])
            return np.zeros(1)

def test_concept_gen(model, tok, max_length, prompts, targets, device):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompts = [prompt + ' ' for prompt in prompts]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    with torch.no_grad():
        pre_edit_outputs = model.generate(
            input_ids=prompt_tok['input_ids'].to(f"cuda:{device}"),
            attention_mask=prompt_tok['attention_mask'].to(f"cuda:{device}"),            
            # input_ids=prompt_tok['input_ids'].to("cpu"),
            # attention_mask=prompt_tok['attention_mask'].to("cpu"),
            max_new_tokens=40,
            pad_token_id=tok.eos_token_id
        )

        model_response = [tok.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        answer = model_response[0][len(prompts[0]):]
        return answer

#    DS, DG_onlyQ, DG_otherA, DG_otherQ, DG_otherAQ = test_safety_gen(model, tok, batch, device, max_tokens, max_output_tokens,prompt, request, **generate_kwargs)
def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """提取 n-grams 的函数。

    Args:
        tokens (List[str]): 输入的 token 列表。
        n (int): n-grams 的大小。

    Returns:
        List[Tuple[str, ...]]: 返回的 n-grams 列表。
    """
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
def load_toxic_layer_from_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 假设 data 是一个整数列表
            print(f"加载的 JSON 数据: {data}")  # 打印加载的列表数据
            
            # 获取最新的 safe_layer（最后一个元素）
            if isinstance(data, list) and len(data) > 0:
                return data[-1]  # 直接返回最后一个元素
            else:
                print("数据为空或格式不正确")
                return None
    except json.JSONDecodeError as e:
        print(f"JSON 解码错误: {e}")
    except Exception as e:
        print(f"发生其他错误: {e}")
    return None
def test_safety_gen(
        dscd,
        save_path,
        model, 
        tokenizer, 
        test_prompt, 
        cuda,
        max_tokens = 1024, 
        max_output_tokens=500,
        toxic_layer = None,
        max_new_tokens=256, 
        top_p=0.95, 
        top_k=0,
        temperature=0.7,
        mature_layer=None,
        premature_layer=None,

        candidate_premature_layers=[],
        mode='DSCD',
        verbose=True, 
        remove_stop_words=False,
        relative_top=0.1, 
        **kwargs):
    tokenizer.padding_side = 'left'
    # stopping_criteria = StoppingCriteriaList()

    # if input_tokens (at least 1024) + output_tokens (at least 600) < 1624, truncate the input length (from right to left, as harmful questions typically appear on the right)
    if max_tokens < 1024:
        only_response = []
        for item in test_prompt:
            input = tokenizer([item,], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{cuda}")
            input_ids = tokenizer([item,], return_tensors="pt", padding=True, truncation=True).input_ids.to(f"cuda:{cuda}")
            # input = tokenizer([item,], return_tensors="pt", padding=True, truncation=True).to("cpu")
            # input_ids = tokenizer([item,], return_tensors="pt", padding=True, truncation=True).input_ids.to("cpu")
            max_len = input_ids.shape[-1] + max_new_tokens

            if input["input_ids"].size(-1) > max_tokens-max_output_tokens:
                input = {k: v[:, -(max_tokens - max_output_tokens):] for k, v in input.items()}
                input_ids = {k: v[:, -(max_tokens - max_output_tokens):] for k, v in input.input_ids.items()}############?这个有什么用？
            with torch.no_grad():
                # outputs = model.generate(**input, max_new_tokens=max_output_tokens)
                ##############################################################################################
                # mode = 'dola'
                # assert mature_layer is not None, "mature_layer must be specified"
                # assert candidate_premature_layers is not None, "candidate_premature_layers must be specified"
                # toxic_layer = _locate_toxic_layer(model, tokenizer, request, hparams)
                # toxic_layer = toxic_layer[0]+1
                # print("toxic_layer:{}".format(toxic_layer))
                # outputs = self.model.generate(input_ids, max_length=max_len, num_return_sequences=1,
                #                         output_scores=True, return_dict_in_generate=True, dola_decoding=True,
                #                         top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, relative_top=relative_top, 
                #                         mature_layer=mature_layer, toxic_layer = toxic_layer[0],premature_layer=None, candidate_premature_layers=candidate_premature_layers, **kwargs,)
                # kwargs = dict(do_sample=True, max_new_tokens=1024, repetition_penalty=1.2, mode=mode, remove_stop_words=True, eos_token_id=eos_token_id)
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_output_tokens,                
                    num_return_sequences=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    dscd_decoding=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    relative_top=relative_top,
                    mature_layer=mature_layer,
                    toxic_layer=toxic_layer,
                    premature_layer=None,
                    candidate_premature_layers=candidate_premature_layers,
                    eos_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2,
                    **kwargs
                )               
     
                
                
                
                
                
                
                
                ###############################################################################################
                texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                texts = texts[0]
            if input["input_ids"].size(-1) > max_tokens-max_output_tokens:
                max_overlap_len = min(len(item), len(texts))
                overlap = next((item[-i:] for i in range(max_overlap_len, 0, -1) if item[-i:] == texts[:i]), "")
            else:
                overlap = item
            only_response.append(texts[len(overlap)+1:].lstrip())
        return only_response
    else:
        # # input_ids = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).input_ids.to("cpu")



        with torch.no_grad():
            ####################################################################################################################
        #     outputs = model.generate(
        #     **input,
        #     max_length=max_len,
        #     num_return_sequences=1,
        #     output_scores=True,
        #     return_dict_in_generate=True,
        #     dola_decoding=True,
        #     top_p=top_p,
        #     top_k=top_k,
        #     temperature=temperature,
        #     relative_top=relative_top,
        #     mature_layer=mature_layer,
        #     toxic_layer=toxic_layer[0],
        #     premature_layer=None,
        #     candidate_premature_layers=candidate_premature_layers,
        #     eos_token_id=tokenizer.eos_token_id,
        #     no_repeat_ngram_size=2,
        #     **kwargs
        # ) 
        
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            # test_prompt="The protagonist was angry and shouted: What the f"
            # test_prompt="So I am starting to think she is full of 's"
            # test_prompt="She's seriously talented, though her ideas are often s"
            # test_prompt='What the F'
            input = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{0}")
            # save_path="/mnt/sdb/zjk/EasyEdit_dola/Qwen_result/s-t+b"
            # save_path=save_path
            if dscd is True:
                # input = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{0}")
                # max_len = input_ids.shape[-1] +max_new_tokens
                # outputs,safe_layer,toxic = model.generate(
                #     dola,
                #     **input,
                #     max_new_tokens=max_output_tokens,
                #     num_return_sequences=1,
                #     output_scores=True,
                #     dola_decoding=True,
                #     top_p=top_p,
                #     top_k=top_k,
                #     temperature=temperature,
                #     relative_top=relative_top,
                #     mature_layer=mature_layer,
                #     toxic_layer=toxic_layer,
                #     premature_layer=None,
                #     candidate_premature_layers=candidate_premature_layers,
                #     eos_token_id=tokenizer.eos_token_id,
                #     **kwargs
                # )   
                # outputs,safe_layer,toxic,h = model.generate(
                #     dola,
                #     **input,
                #     max_new_tokens=max_output_tokens,
                #     relative_top=relative_top,
                #     mature_layer=mature_layer,
                #     toxic_layer=toxic_layer,
                #     premature_layer=None,
                #     candidate_premature_layers=candidate_premature_layers,
                #     do_sample=False,  # 开启采样模式
                #     output_scores=True,  # 确保每步的 logits 都返回
                #     return_dict_in_generate=True  # 返回包含 logits 的  字典
                # )
                outputs,safe_layer,toxic,h = model.generate(
                    dscd,
                    **input,
                    max_new_tokens=max_output_tokens,
                    relative_top=relative_top,
                    mature_layer=mature_layer,
                    toxic_layer=toxic_layer,
                    premature_layer=None,
                    candidate_premature_layers=candidate_premature_layers,
                    do_sample=False,
                    # output_scores=True,
                    # return_dict_in_generate=True,
                )

                save_toxic_layer_to_json(safe_layer, save_path+"/safe_layer_b-S+T.json")
                save_toxic_layer_to_json(toxic, save_path+"/toxic1.json")
                save_toxic_layer_to_json(h, save_path+"/hallucination1.json")
                
                #                  # 获取所有 logits
                # # Extract the logits for each step
                # logits = outputs.scores  # List of logits for each step
                # input_ids = outputs.sequences  # Generated input IDs

                #                 # For the first token (after the prompt), calculate probabilities and find the token corresponding to "milk"
                # # 从 logits 中获取所有 token 的概率分布
                # first_token_logits = logits[0]  # 第一步预测的 logits
                # first_token_probs = torch.softmax(first_token_logits, dim=-1)  # 计算 softmax 得到概率

                # # 获取 top-k 概率较高的 token 以及它们的概率
                # top_k = 5  # 获取 top 5 的候选 token
                # top_k_probs, top_k_indices = torch.topk(first_token_probs, top_k, dim=-1)

                # # 输出 top-k token 的 ID、对应 token 和概率
                # print(f"Top {top_k} tokens and their probabilities:")
                # for i in range(top_k):
                #     token_id = top_k_indices[0, i].item()
                #     token_word = tokenizer.convert_ids_to_tokens(token_id)
                #     token_prob = top_k_probs[0, i].item()
                #     print(f"Token ID: {token_id}, Token: {token_word}, Probability: {token_prob:.4f}")

                # # 打印 "milk" 的概率
                # milk_token = tokenizer.encode("**t", add_special_tokens=False)[0]  # 获取 "milk" 的 token ID
                # milk_token_prob = first_token_probs[0, milk_token].item()  # 获取 "milk" 的概率
                # print(f"Probability of 'shit': {milk_token_prob:.4f}")
                # # Decode the entire generated sequence
                # texts = [tokenizer.decode(output, skip_special_tokens=True) for output in input_ids]
                # only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]

                # # Output the full responses
                # print("Generated Responses:")
                # for response in only_response:
                #     print(response)
                # outputs = model.generate(dola,**input, max_new_tokens=max_output_tokens,do_sample=False)

                # premature_layer_dist = outputs.premature_layer_dist
                # presafe_layer_dist = outputs.presafe_layer_dist
                # sequences, scores = outputs.sequences, outputs.scores

                # # skip the tokens in the input prompt
                # gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
                # gen_arr = gen_sequences.cpu().numpy()

                # output_strs = tokenizer.decode(gen_sequences, skip_special_tokens=True)
                # print(len(test_prompt), len(output_strs))

                # ######################################################################################################################
                # outputs = model.generate(dola,**input, max_new_tokens=max_output_tokens)

                texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]
                # texts = [tokenizer.decode(output, skip_special_tokens=True) for output in sequences]
                # print(texts)
                # only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]                    

            else:
                # DINM method: Use standard transformers generate (no DSCD decoding)
                # The edited_model is a standard LlamaForCausalLM, not DSCDLlamaForCausalLM
                outputs,safe_layer,toxic,h = model.generate(
                    dscd,
                    **input,
                    max_new_tokens=max_output_tokens,
                    relative_top=relative_top,
                    mature_layer=mature_layer,
                    toxic_layer=toxic_layer,
                    premature_layer=None,
                    candidate_premature_layers=candidate_premature_layers,
                    do_sample=False,
                    # output_scores=True,
                    # return_dict_in_generate=True,
                )
                
                # Save placeholder values for consistency with DSCD
                save_toxic_layer_to_json(safe_layer, save_path+"/safe_layer_2.json")
                save_toxic_layer_to_json(toxic, save_path+"/toxic2.json")
                save_toxic_layer_to_json(h, save_path+"/hallucination2.json")
                # outputs = model.generate(dola,**input, max_new_tokens=max_output_tokens,do_sample=False)

                # premature_layer_dist = outputs.premature_layer_dist
                # presafe_layer_dist = outputs.presafe_layer_dist
                # sequences, scores = outputs.sequences, outputs.scores

                # skip the tokens in the input prompt
                # gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
                # gen_arr = gen_sequences.cpu().numpy()

                # output_strs = tokenizer.decode(gen_sequences, skip_special_tokens=True)
                # print(len(test_prompt), len(output_strs))

                # ######################################################################################################################
                # texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

                # only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]
                
                
                
                # outputs = model.generate(dola,**input, max_new_tokens=max_output_tokens,do_sample=False)
                texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]
                # outputs = model.generate(
                #     dola,
                #     **input,
                #     max_new_tokens=max_output_tokens,
                #     num_return_sequences=1,
                #     output_scores=True,
                #     dola_decoding=True,
                #     top_p=top_p,
                #     top_k=top_k,
                #     temperature=temperature,
                #     relative_top=relative_top,
                #     mature_layer=mature_layer,
                #     toxic_layer=toxic_layer[0],
                #     premature_layer=None,
                #     candidate_premature_layers=candidate_premature_layers,
                #     eos_token_id=tokenizer.eos_token_id,
                #     no_repeat_ngram_size=2,
                #     **kwargs
                # )    
                # premature_layer_dist = outputs.premature_layer_dist
                # presafe_layer_dist = outputs.presafe_layer_dist
                # sequences, scores = outputs.sequences, outputs.scores

                # skip the tokens in the input prompt
                # gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
                # gen_arr = gen_sequences.cpu().numpy()

                # output_strs = tokenizer.decode(gen_sequences, skip_special_tokens=True)
                # print(len(test_prompt), len(output_strs))

                # ######################################################################################################################
                # texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                # only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]
            return only_response