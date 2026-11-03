import argparse
import os

import torch
from transformers import AutoTokenizer, Gemma2ForCausalLM

from clm_moerging_strategies_eval import moerge
from utils.reasoning_args_manager import GEMMA_MODEL_MAP, TASK_MAP
from utils.utils import get_task, load_moerged_model, get_context

from datasets import load_dataset


os.environ["HF_ALLOW_CODE_EVAL"] = "1"


def set_dume_routers(model, use_test_distribution, domains, max_num_samples, batch_size, tokenizer):

    model.set_rr_gate_mode(True)

    task_dicts = {}
    if use_test_distribution:
        print("Using test distribution for DUME statistics collection.")
        tasks = [TASK_MAP[domain] for domain in domains]
        for task_name in tasks:
            task_dict, _ = get_task(task_name)
            task_dicts[task_name] = task_dict
    else:
        print("Using train distribution for DUME statistics collection.")
        task_dicts = {
            "hkust-nlp/dart-math-hard": load_dataset("hkust-nlp/dart-math-hard")['train'],
            "CohereLabs/aya_dataset": load_dataset("CohereLabs/aya_dataset")['train'],
            "ise-uiuc/Magicoder-OSS-Instruct-75K": load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")['train'],
            "allenai/tulu-3-sft-personas-instruction-following": load_dataset("allenai/tulu-3-sft-personas-instruction-following")['train']
        }
        
    print("Collecting DUME statistics...")
    with torch.no_grad():
        for expert_id, (task_name, task) in enumerate(task_dicts.items()):
            print(f"Collecting DUME stats for domain {task_name}...")
            if use_test_distribution:
                if task_name != 'm_arc':
                    try:
                        dataset = list(task[task_name].dataset['train'])[:max_num_samples]  # type: ignore
                    except KeyError:
                        dataset = list(task[task_name].dataset['test'])[:max_num_samples]  # type: ignore
                else:
                    remaining_samples = max_num_samples
                    dataset = []
                    while remaining_samples > 0:
                        num_samples_per_arc = max(1, int(remaining_samples / len(task.keys())))  # type: ignore
                        for arc_task_name in task.keys():  # type: ignore
                            dataset.extend(list(task[arc_task_name].dataset['train'])[:num_samples_per_arc])
                            remaining_samples -= num_samples_per_arc
                            if remaining_samples <= 0:
                                break
            else:
                dataset = list(task)[:max_num_samples]

            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                context = get_context(task_name, batch)
                input_ids = tokenizer(context, return_tensors='pt', padding=True, truncation=True).input_ids.cuda()
                model(input_ids=input_ids, expert_id=expert_id)
    
    model.set_rr_gate_mode(False)


def main():

    parser = argparse.ArgumentParser(description="Evaluate MergeBench models on various tasks")
    parser.add_argument("--model-type", type=str, choices=["llama", "gemma"], default="llama", help="Whether to evaluate the LLaMA-based model or the Gemma-based model")
    parser.add_argument("--num_dume_tokens", type=int, default=1024, help="Context dimension for routing")
    parser.add_argument("--temp", type=float, default=1.0, help="Softmax temperature for routing")
    parser.add_argument("--_lambda", type=float, default=1e-2, help="Tikhonov regularization parameter")
    parser.add_argument("--k", type=int, default=1, help="K of top-K routing")
    parser.add_argument("--domains", type=str, nargs='+', default=list(TASK_MAP.keys()), help="Domains for average strategy")
    parser.add_argument("--num_parameters", type=float, default=3e9, help="Number of parameters for the base dense model experts")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--max_num_samples', type=int, default=2048, help='Maximum number of samples to use for DUME statistics collection')
    parser.add_argument("--use_test_distribution", type=bool, default=True, help="Whether to use test distribution for DUME statistics collection")
    
    args = parser.parse_args()

    if args.model_type == "llama":
        ckpt_path = "checkpoints/post_training_btx_model_random_gating.pth"
        tokenizer = AutoTokenizer.from_pretrained("MergeBench/Llama-3.2-3B_coding")  # Any domain works for tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        model, moe_experts, config = load_moerged_model(ckpt_path, args.num_parameters, args.k, args.temp, args._lambda, args.num_dume_tokens, len(TASK_MAP), args.model_type)
        model.cuda()  # type: ignore
    elif args.model_type == "gemma":
        ckpt_path = "checkpoints/post_training_btx_model_random_gating_gemma.pth"
        tokenizer = AutoTokenizer.from_pretrained("MergeBench/gemma-2-9b_coding")  # Any domain works for tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        configs = []
        state_dicts = []
        model_names = list(GEMMA_MODEL_MAP.values())
        for mn in model_names:
            tmp_model = Gemma2ForCausalLM.from_pretrained(mn, torch_dtype=torch.bfloat16, device_map="cpu")
            configs.append(tmp_model.config)
            model_state_dict = tmp_model.state_dict()
            state_dicts.append(model_state_dict)
        config = configs[-1]
        model, moe_experts = moerge(state_dicts, device, config=config, num_parameters=args.num_parameters,
                                    domains=args.domains, k=args.k, temp=args.temp, _lambda=args._lambda,
                                    context_dim=args.num_dume_tokens, return_moe_experts=True, model_type=args.model_type)  
        model = model.to(torch.bfloat16)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not supported")
    
    set_dume_routers(model, args.use_test_distribution, args.domains, args.max_num_samples, args.batch_size, tokenizer)    

    if args.model_type == "llama":
        str_domains = f"_domains={'-'.join(args.domains)}" if len(args.domains) != len(TASK_MAP) else ''
        str_model_type = f'_{args.model_type}' if args.model_type == 'gemma' else ''
        str_ood = '_OOD' if not args.use_test_distribution else ''
        new_ckpt_path = f"checkpoints/reasoning_DUME{str_model_type}{str_ood}_k={args.k}_temp={args.temp}_λ={args._lambda}_cdim={args.num_dume_tokens}{str_domains}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'moe_experts_state_dicts': [[expert.state_dict() for expert in experts] for experts in moe_experts],
            'config': config
        }, new_ckpt_path)
    

if __name__ == "__main__":
    main()
