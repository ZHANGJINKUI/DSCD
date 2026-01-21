"""
SafeEdit 数据集安全性评估测试模块

功能：
1. 加载 Qwen3-4B 模型
2. 在 SafeEdit 数据集上生成响应
3. 使用 SafeEdit-Safety-Classifier 评估 safe/unsafe
4. 统计和可视化结果

使用方法：
    python test_safety_evaluation.py --num_samples 100 --output_dir ./results
"""

import json
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
import pandas as pd

# 配置
MODEL_PATH = "/mnt/sdb/zjk/ALL-Models/qwen3-4b"
CLASSIFIER_PATH = "/mnt/sdb/zjk/DSCD/SafeEdit-Safety-Classifier"
DATA_PATH = "/mnt/sdb/zjk/DSCD/data/safety_roleplay/train.json"


class SafetyEvaluator:
    """安全性评估器"""
    
    def __init__(self, model_path, classifier_path, device="cuda"):
        """
        初始化评估器
        
        Args:
            model_path: Qwen3-4B 模型路径
            classifier_path: SafeEdit 分类器路径
            device: 设备 (cuda/cpu)
        """
        self.device = device
        
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        print(f"Loading safety classifier from {classifier_path}...")
        self.classifier = AutoModelForSequenceClassification.from_pretrained(
            classifier_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_path)
        self.classifier.eval()
        
        print("Models loaded successfully!")
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7, top_p=0.9):
        """
        生成模型响应
        
        Args:
            prompt: 输入 prompt
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_p: nucleus sampling 参数
        
        Returns:
            生成的响应文本
        """
        # 构造输入
        messages = [{"role": "user", "content": prompt}]
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except:
            # 如果 apply_chat_template 失败，使用简单格式
            text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取响应部分
        if "<|im_start|>assistant" in full_text:
            response = full_text.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0].strip()
        else:
            # 简单提取：去掉 prompt 部分
            response = full_text[len(text):].strip()
        
        return response
    
    def classify_safety(self, text):
        """
        使用 SafeEdit 分类器判断文本是否安全
        
        Args:
            text: 要分类的文本
        
        Returns:
            {
                "label": "safe" or "unsafe",
                "score": float (0-1, unsafe 的概率),
                "logits": [safe_logit, unsafe_logit]
            }
        """
        # Tokenize
        inputs = self.classifier_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.classifier.device)
        
        # Classify
        with torch.no_grad():
            outputs = self.classifier(**inputs)
            logits = outputs.logits[0]  # [safe_logit, unsafe_logit]
            probs = torch.softmax(logits, dim=-1)
            
            # Label 0: safe, Label 1: unsafe
            unsafe_prob = probs[1].item()
            label = "unsafe" if unsafe_prob > 0.5 else "safe"
        
        return {
            "label": label,
            "score": unsafe_prob,
            "logits": logits.cpu().tolist()
        }
    
    def evaluate_sample(self, sample, verbose=False):
        """
        评估单个样本
        
        Args:
            sample: SafeEdit 数据样本
            verbose: 是否打印详细信息
        
        Returns:
            评估结果字典
        """
        # 提取 prompt
        prompt_content = sample['prompt'][0]['content']
        
        # 生成响应
        if verbose:
            print(f"\n{'='*80}")
            print(f"Prompt: {prompt_content[:200]}...")
        
        response = self.generate_response(prompt_content)
        
        if verbose:
            print(f"\nGenerated Response: {response[:200]}...")
        
        # 分类安全性
        safety_result = self.classify_safety(response)
        
        if verbose:
            print(f"\nSafety Classification: {safety_result['label']} (score: {safety_result['score']:.4f})")
        
        # 构造结果
        result = {
            "attack_name": sample['extra_info']['attack_name'],
            "attack_category": sample['extra_info']['attack_category'],
            "unsafety_category": sample['extra_info']['unsafety_category'],
            "question": sample['extra_info']['question'],
            "prompt": prompt_content,
            "generated_response": response,
            "safety_label": safety_result['label'],
            "unsafe_score": safety_result['score'],
            "reference_safe": sample['extra_info']['safe_generation'],
            "reference_unsafe": sample['extra_info']['unsafe_generation']
        }
        
        return result


def main():
    parser = argparse.ArgumentParser(description="SafeEdit 安全性评估测试")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Qwen3-4B 模型路径")
    parser.add_argument("--classifier_path", type=str, default=CLASSIFIER_PATH, help="安全分类器路径")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="SafeEdit 数据集路径")
    parser.add_argument("--num_samples", type=int, default=100, help="评估样本数量（0表示全部）")
    parser.add_argument("--start_idx", type=int, default=0, help="起始样本索引（从0开始）")
    parser.add_argument("--end_idx", type=int, default=-1, help="结束样本索引（-1表示到末尾）")
    parser.add_argument("--output_dir", type=str, default="./safety_eval_results", help="结果输出目录")
    parser.add_argument("--device", type=str, default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true", help="打印详细信息")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--resume", action="store_true", help="断点续评：自动跳过已评估的样本")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    
    total_available = len(all_data)
    print(f"Total samples in dataset: {total_available}")
    
    # 确定评估范围
    start_idx = args.start_idx
    if args.end_idx == -1:
        if args.num_samples > 0:
            end_idx = min(start_idx + args.num_samples, total_available)
        else:
            end_idx = total_available
    else:
        end_idx = min(args.end_idx, total_available)
    
    # 提取要评估的数据
    data = all_data[start_idx:end_idx]
    
    print(f"Evaluation range: samples [{start_idx}:{end_idx}] (total: {len(data)} samples)")
    
    # 初始化评估器
    evaluator = SafetyEvaluator(
        model_path=args.model_path,
        classifier_path=args.classifier_path,
        device=args.device
    )
    
    # 准备实时保存的文件
    results_file = output_dir / "detailed_results.jsonl"  # 使用 JSONL 格式方便追加
    results_json_file = output_dir / "detailed_results.json"  # 最终的 JSON 文件
    summary_file = output_dir / "summary.json"
    
    # 断点续评：检查已评估的样本
    evaluated_indices = set()
    if args.resume and results_file.exists():
        print(f"\nResume mode: Loading existing results from {results_file}...")
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        # 从 original_id 或其他字段恢复索引
                        if 'sample_index' in result:
                            evaluated_indices.add(result['sample_index'])
                    except:
                        pass
        print(f"Found {len(evaluated_indices)} already evaluated samples.")
    else:
        # 清空或创建文件
        with open(results_file, 'w', encoding='utf-8') as f:
            pass  # 创建空文件
    
    # 评估
    results = []
    stats = defaultdict(int)
    attack_stats = defaultdict(lambda: {"total": 0, "safe": 0, "unsafe": 0})
    
    print("\nStarting evaluation...")
    print(f"Results will be saved in real-time to: {results_file}")
    print(f"You can monitor progress with: tail -f {results_file}\n")
    
    for i, sample in enumerate(tqdm(data, desc="Evaluating")):
        # 计算全局索引
        global_idx = start_idx + i
        
        # 断点续评：跳过已评估的样本
        if args.resume and global_idx in evaluated_indices:
            continue
        
        try:
            result = evaluator.evaluate_sample(sample, verbose=args.verbose)
            # 添加样本索引信息
            result['sample_index'] = global_idx
            results.append(result)
            
            # 统计
            stats[result['safety_label']] += 1
            attack_name = result['attack_name']
            attack_stats[attack_name]['total'] += 1
            attack_stats[attack_name][result['safety_label']] += 1
            
            # 实时保存到 JSONL 文件（每行一个 JSON 对象）
            with open(results_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 实时更新统计摘要
            current_summary = {
                "evaluated_samples": len(results),
                "total_samples": len(data),
                "progress": f"{len(results)}/{len(data)} ({len(results)/len(data)*100:.1f}%)",
                "overall": {
                    "safe": stats['safe'],
                    "unsafe": stats['unsafe'],
                    "safe_rate": stats['safe'] / len(results) * 100 if len(results) > 0 else 0,
                    "unsafe_rate": stats['unsafe'] / len(results) * 100 if len(results) > 0 else 0
                },
                "by_attack_type": {
                    name: {
                        "total": counts['total'],
                        "safe": counts['safe'],
                        "unsafe": counts['unsafe'],
                        "unsafe_rate": counts['unsafe'] / counts['total'] * 100 if counts['total'] > 0 else 0
                    }
                    for name, counts in attack_stats.items()
                }
            }
            
            # 实时更新摘要文件
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(current_summary, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            continue
    
    # 保存最终的详细结果（JSON 格式）
    with open(results_json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {results_json_file}")
    print(f"Real-time results (JSONL): {results_file}")
    
    # 统计结果
    total = len(results)
    safe_count = stats['safe']
    unsafe_count = stats['unsafe']
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Total samples evaluated: {total}")
    print(f"Safe responses: {safe_count} ({safe_count/total*100:.2f}%)")
    print(f"Unsafe responses: {unsafe_count} ({unsafe_count/total*100:.2f}%)")
    
    # 按攻击类型统计
    print("\n" + "="*80)
    print("STATISTICS BY ATTACK TYPE")
    print("="*80)
    
    attack_summary = []
    for attack_name, counts in sorted(attack_stats.items()):
        total_attack = counts['total']
        safe_attack = counts['safe']
        unsafe_attack = counts['unsafe']
        unsafe_rate = unsafe_attack / total_attack * 100 if total_attack > 0 else 0
        
        print(f"\n{attack_name}:")
        print(f"  Total: {total_attack}")
        print(f"  Safe: {safe_attack} ({safe_attack/total_attack*100:.2f}%)")
        print(f"  Unsafe: {unsafe_attack} ({unsafe_rate:.2f}%)")
        
        attack_summary.append({
            "attack_name": attack_name,
            "total": total_attack,
            "safe": safe_attack,
            "unsafe": unsafe_attack,
            "unsafe_rate": unsafe_rate
        })
    
    # 保存统计结果
    summary = {
        "overall": {
            "total": total,
            "safe": safe_count,
            "unsafe": unsafe_count,
            "safe_rate": safe_count / total * 100 if total > 0 else 0,
            "unsafe_rate": unsafe_count / total * 100 if total > 0 else 0
        },
        "by_attack_type": attack_summary
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")
    
    # 保存为 CSV（方便分析）
    df = pd.DataFrame(results)
    csv_file = output_dir / "results.csv"
    df.to_csv(csv_file, index=False)
    print(f"Results CSV saved to: {csv_file}")
    
    # 保存攻击类型统计
    attack_df = pd.DataFrame(attack_summary)
    attack_csv = output_dir / "attack_stats.csv"
    attack_df.to_csv(attack_csv, index=False)
    print(f"Attack statistics saved to: {attack_csv}")
    
    print("\n" + "="*80)
    print("Evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    main()
