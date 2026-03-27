import os
import sys
import random
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, LlamaForCausalLM, set_seed
from datasets import load_dataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from utils.distributed import distributed_init
from torch.distributed.fsdp.api import MixedPrecision, ShardingStrategy

from models.llama_experts import get_llama_expert
from models.llama import LLamaWrapperEval
from utils.reasoning_args_manager import get_args, TASK_MAP
from utils.import_hooks import CodeEvalImportHook
from utils.utils import align_and_stack, get_task, load_moerged_model, get_context
from utils.optimizer import get_training_strategy
from clm_moerging_strategies_eval import moerge
from clm_trainer import train

sys.path.insert(0, '/home/gensyn/persistent/DetGatingMoE-exp/utils')
sys.meta_path.insert(0, CodeEvalImportHook())

from lm_eval.evaluator import evaluate
from utils.execute import set_total_tests


os.environ["HF_ALLOW_CODE_EVAL"] = "1"



def main():

    args, model_name, task_name = get_args()
    set_seed(args.seed)

    print()
    print("*" * 100)
    print(f"Evaluating model: {model_name} on task: {task_name} with strategy: {args.strategy}")
    print("*" * 100)
    print()
    
    if 'expert' in args.strategy:

        assert isinstance(model_name, str), "Model name must be a single string for expert strategies"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name)
        model.cuda()  # type: ignore
        model_state_dict = model.state_dict()
        model_state_dict = {f'_model.{k}': v for k, v in model_state_dict.items()}

        model_custom = get_llama_expert(3e9, config=model.config)[0]
        model_custom.cuda()
        model_custom.load_state_dict(model_state_dict, strict=True)
        model_custom = LLamaWrapperEval(model_custom, tokenizer, device="cuda")

    elif args.strategy == "model_averaging":

        tokenizer = AutoTokenizer.from_pretrained(model_name[-1])
        configs = []

        state_dicts = []
        for mn in model_name:
            model = LlamaForCausalLM.from_pretrained(mn)
            configs.append(model.config)
            model_state_dict = model.state_dict()
            model_state_dict = {f'_model.{k}': v for k, v in model_state_dict.items()}
            state_dicts.append(model_state_dict)
        
        merged_state_dict = {}

        for k in state_dicts[0].keys():
            tensors = [sd[k] for sd in state_dicts]
            if len(set(t.shape for t in tensors)) == 1:
                merged_state_dict[k] = torch.mean(torch.stack(tensors), dim=0)
            else:
                print(f"Warning: Aligning tensors for key: {k}, shapes: {[t.shape for t in tensors]}")
                merged_state_dict[k] = torch.mean(align_and_stack(tensors), dim=0)
        
        del state_dicts
        
        model_custom = get_llama_expert(3e9, config=configs[-1])[0]
        model_custom.cuda()
        model_custom.load_state_dict(merged_state_dict, strict=True)
        model_custom = LLamaWrapperEval(model_custom, tokenizer, device="cuda")

    elif args.strategy == "oracle" or args.strategy == "random_routing":
        
        ckpt_path = "checkpoints/reasoning_random_routing.pth"

        if not os.path.exists(ckpt_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            configs = []
            state_dicts = []
            for mn in model_name:
                model = LlamaForCausalLM.from_pretrained(mn)
                configs.append(model.config)
                model_state_dict = model.state_dict()
                model_state_dict = {f'_model.{k}': v for k, v in model_state_dict.items()}
                state_dicts.append(model_state_dict)

            model_custom, moe_experts = moerge(state_dicts, device, config=configs[-1], num_parameters=args.num_parameters,
                                                             domains=args.domains, k=args.k, temp=args.temp, _lambda=args._lambda,
                                                             context_dim=args.num_dume_tokens, return_moe_experts=True)  # type: ignore
            torch.save({
                'model_state_dict': model_custom.state_dict(),
                'moe_experts_state_dicts': [[expert.state_dict() for expert in experts] for experts in moe_experts],
                'config': configs[-1]
            }, ckpt_path)

        else:
            model_custom, moe_experts, _ = load_moerged_model(ckpt_path, args.num_parameters, args.k, args.temp, args._lambda, args.num_dume_tokens, len(TASK_MAP))

        tokenizer = AutoTokenizer.from_pretrained(model_name[-1])
        model_custom = LLamaWrapperEval(model_custom, tokenizer, device="cuda",
                                        expert_id=list(TASK_MAP.values()).index(task_name) if args.strategy == "oracle" else None)

    elif args.strategy == "DUME":
        str_domains = f"_domains={'-'.join(args.domains)}" if len(args.domains) != len(TASK_MAP) else ''
        ckpt_path = f"checkpoints/reasoning_DUME{'_OOD' if not args.use_test_distribution else ''}_k={args.k}_temp={args.temp}_λ={args._lambda}_cdim={args.num_dume_tokens}{str_domains}.pth"
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist. Please run reasoning_generate_dume_params.py first."
        model_custom, moe_experts, _ = load_moerged_model(ckpt_path, args.num_parameters, args.k, args.temp, args._lambda, args.num_dume_tokens, len(TASK_MAP))
        tokenizer = AutoTokenizer.from_pretrained(model_name[-1])
        model_custom = LLamaWrapperEval(model_custom, tokenizer, device="cuda", expert_id=None)
    
    elif args.strategy == "BTX" or args.strategy == "DUMEplus":

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        reasoning_ckpt_path = f"checkpoints/{'OOD_' if not args.use_test_distribution else ''}reasoning_{args.strategy}_lr={args.lr}_alpha={args.alpha}_steps_{args.num_btx_steps}_seed={args.seed}_model.pth"
        tokenizer = AutoTokenizer.from_pretrained(model_name[-1])
        tokenizer.pad_token = tokenizer.eos_token
        
        if not os.path.exists(reasoning_ckpt_path):

            print("MoErging...")

            ckpt_path = "checkpoints/reasoning_random_routing.pth"

            if not os.path.exists(ckpt_path):
                
                configs = []
                state_dicts = []
                for mn in model_name:
                    model = LlamaForCausalLM.from_pretrained(mn)
                    configs.append(model.config)
                    model_state_dict = model.state_dict()
                    model_state_dict = {f'_model.{k}': v for k, v in model_state_dict.items()}
                    state_dicts.append(model_state_dict)

                model_custom, moe_experts = moerge(state_dicts, device, config=configs[-1], num_parameters=args.num_parameters,
                                                   domains=args.domains, k=args.k, temp=args.temp, _lambda=args._lambda,
                                                   context_dim=args.num_dume_tokens, return_moe_experts=True)  # type: ignore
                
                config = configs[-1]
                torch.save({
                    'model_state_dict': model_custom.state_dict(),
                    'moe_experts_state_dicts': [[expert.state_dict() for expert in experts] for experts in moe_experts],
                    'config': config
                }, ckpt_path)

            else:
                model_custom, moe_experts, config = load_moerged_model(ckpt_path, args.num_parameters, args.k, args.temp, args._lambda, args.num_dume_tokens, len(TASK_MAP))
 
            model_custom.to(device)

            print("Merging datasets...")

            task_dicts = {}
            if args.use_test_distribution:
                print("Using test distribution.")
                tasks = [TASK_MAP[domain] for domain in args.domains]
                for tn in tasks:
                    task_dict, _ = get_task(tn)
                    task_dicts[tn] = task_dict
            else:
                print("Using train distribution.")
                task_dicts = {
                    "hkust-nlp/dart-math-hard": load_dataset("hkust-nlp/dart-math-hard")['train'],
                    "CohereLabs/aya_dataset": load_dataset("CohereLabs/aya_dataset")['train'],
                    "ise-uiuc/Magicoder-OSS-Instruct-75K": load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")['train'],
                    "allenai/tulu-3-sft-personas-instruction-following": load_dataset("allenai/tulu-3-sft-personas-instruction-following")['train']
                }

            dataloader = []
            with torch.no_grad():
                full_datasets = {}
                for tn, task in tqdm(task_dicts.items()):
                    print(f"Processing task: {tn}...")
                    if args.use_test_distribution:
                        if tn != 'm_arc':
                            try:
                                dataset = list(task[tn].dataset['train'])  # type: ignore
                            except KeyError:
                                dataset = list(task[tn].dataset['test'])  # type: ignore

                            if len(dataset) < args.max_num_samples:
                                original_dataset = dataset.copy()
                                while len(dataset) < args.max_num_samples:
                                    remaining_needed = args.max_num_samples - len(dataset)
                                    if remaining_needed >= len(original_dataset):
                                        dataset.extend(original_dataset)
                                    else:
                                        dataset.extend(original_dataset[:remaining_needed])
                                        break

                        else:
                            remaining_samples = args.max_num_samples
                            dataset = []
                            while remaining_samples > 0:
                                num_samples_per_arc = max(1, int(remaining_samples / len(task.keys())))  # type: ignore
                                for arc_task_name in task.keys():  # type: ignore
                                    dataset.extend(list(task[arc_task_name].dataset['train'])[:num_samples_per_arc])
                                    remaining_samples -= num_samples_per_arc
                                    if remaining_samples <= 0:
                                        break
                        random.shuffle(dataset)
                        full_datasets[tn] = dataset
                    else:
                        full_datasets[tn] = list(task)
                
                min_ds_len = min([len(v) for v in full_datasets.values()])
                if min_ds_len < args.max_num_samples:
                    print(f"Truncating datasets to minimum length: {min_ds_len}")
                datasets = {k: v[:min(args.max_num_samples, min_ds_len)] for k, v in full_datasets.items()}

                for tn, dataset in tqdm(datasets.items()):
                    for i in range(0, len(dataset), args.batch_size):
                        batch = dataset[i:i+args.batch_size]
                        context = get_context(tn, batch)
                        input_ids = tokenizer(context, return_tensors='pt', padding=True, truncation=True).input_ids.cuda()
                        labels = input_ids.clone()
                        labels = torch.cat([labels[:, 1:], torch.full((labels.size(0), 1), tokenizer.pad_token_id, device=labels.device)], dim=1)
                        labels[labels == tokenizer.pad_token_id] = -100  # Set padding tokens to -100 so they're ignored in loss calculation
                        
                        dataloader.append({"input_ids": input_ids, "label": labels, "task_id": list(task_dicts.keys()).index(tn)})
            
            random.shuffle(dataloader)

            if args.strategy == "DUMEplus":
                model_custom.set_rr_gate_mode(True)
                print("Collecting DUME gate statistics...")
                with torch.no_grad():
                    for i, batch in tqdm(enumerate(dataloader)):
                        input_ids = batch["input_ids"].to(device, non_blocking=True)
                        model_custom(input_ids=input_ids, expert_id=batch["task_id"])
                print("Done.")
                model_custom.set_rr_gate_mode(False)

            # Freeze all layers except those with "gate" in their name
            for name, param in model_custom.named_parameters():
                param.requires_grad = ("gate" in name)

            loss_fn, optimizer, scheduler = get_training_strategy(model_custom, args.lr, warmup_steps=10, num_steps=args.num_btx_steps)

            device, world_size, amp_dtype, is_main = distributed_init()
            use_fsdp = world_size > 1 and torch.cuda.is_available()
            if use_fsdp:
                mp_conf = MixedPrecision(param_dtype=amp_dtype, reduce_dtype=amp_dtype, buffer_dtype=amp_dtype)
                model_custom = FSDP(
                    model_custom,
                    device_id=device,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    mixed_precision=mp_conf,
                    use_orig_params=True,
                )

            print(f"\nTraining {args.strategy} model...")
            model_custom.train()
            train(dataloader, None, model_custom, loss_fn, optimizer, scheduler, device, amp_dtype, use_fsdp,
                num_steps=args.num_btx_steps, num_val_steps=1, grad_accum_steps=4, eval_interval=10, is_main=is_main, alpha=args.alpha)
            print(f"{args.strategy} training completed.\n")

            print(f"Saving {args.strategy} model to {reasoning_ckpt_path}...")
            state_dict = model_custom.state_dict()

            torch.save({
                'model_state_dict': state_dict,
                'moe_experts_state_dicts': [[expert.state_dict() for expert in experts] for experts in moe_experts],
                'config': config
            }, reasoning_ckpt_path)
        
        else:
            print(f"Loading {args.strategy} model from {reasoning_ckpt_path}...")
            model_custom, moe_experts, _ = load_moerged_model(reasoning_ckpt_path, args.num_parameters, args.k, args.temp, args._lambda, args.num_dume_tokens, len(TASK_MAP))
            print(f"{args.strategy} model loaded.\n")

        model_custom = LLamaWrapperEval(model_custom, tokenizer, device="cuda", expert_id=None)

    else:
        raise NotImplementedError(f"Strategy {args.strategy} not implemented")

    task_dict, limit = get_task(task_name)

    # Evaluate
    model_custom.model.eval()
    with torch.no_grad():

        if task_name == "humaneval":
            set_total_tests(limit if limit else 164)  # 164 is the total humaneval problems
            
        results = evaluate(
            lm=model_custom,
            task_dict=task_dict,
            limit=limit,
            confirm_run_unsafe_code=True,
        )

    # pprint.pprint(results)
    assert results is not None
    
    str_use_test_distribution = "" if args.use_test_distribution else "_OOD"
    str_cdim = f"_cdim={args.num_dume_tokens}" if 'DUME' in args.strategy else ''
    str_lambda = f"_λ={args._lambda}" if 'DUME' in args.strategy else ''
    str_k = f"_k={args.k}" if 'DUME' in args.strategy else ''
    str_seed = f"_seed={args.seed}"
    str_domains = f"_domains={'-'.join(args.domains)}" if len(args.domains) != len(TASK_MAP) else ''
    str_lr = f"_lr={args.lr}" if args.strategy == 'BTX' else ''
    str_alpha = f"_alpha={args.alpha}" if args.strategy == 'BTX' and args.alpha is not None else ''
    str_steps = f"_steps={args.num_btx_steps}" if args.strategy == 'BTX' else ''
    filename = f"checkpoints/{args.strategy}_{str_use_test_distribution}{task_name}{str_cdim}{str_lambda}{str_k}{str_seed}{str_domains}{str_lr}{str_alpha}{str_steps}_results.txt"
    
    output_lines = []
    
    # MATH
    if task_name == "gsm8k":
        result_text = f"Final result: {round(results['results'][task_name]['exact_match,flexible-extract'] * 100, 2)}"
        print(result_text)
        output_lines.append(result_text)
    elif task_name == 'hendrycks_math':  # TODO: this dataset's evaluation is too strict, it needs a fix. NOT USED
        result_text = f"Final result: {round(results['results'][task_name]['exact_match,none'] * 100, 2)}"
        print(result_text)
        output_lines.append(result_text)
    elif task_name == "mathqa":  # NOT USED
        result_text = f"Final result: {round(results['results'][task_name]['acc_norm,none'] * 100, 2)}"
        print(result_text)
        output_lines.append(result_text)
    # MULTILINGUAL UNDERSTANDING
    elif task_name == "m_mmlu" or task_name == "m_arc":
        avg = 0.0
        if task_name == "m_mmlu":
            metric_name = 'acc,none' 
        elif task_name == "m_arc":
            metric_name = 'acc_norm,none'
        else:
            raise NotImplementedError("Unknown task for multilingual evaluation")
        for k, v in results['results'].items():
            line_text = f"{k}: {round(v[metric_name] * 100, 2)}"
            print(line_text)
            output_lines.append(line_text)
            avg += v[metric_name]
        avg /= len(results['results'])
        final_text = f"Final result (average): {round(avg * 100, 2)}"
        print(final_text)
        output_lines.append(final_text)
    # CODING
    elif task_name == "humaneval":
        result_text = f"Final result: {round(results['results'][task_name]['pass@1,create_test'] * 100, 2)}"
        print(result_text)
        output_lines.append(result_text)
    # INSTRUCTION FOLLOWING
    elif task_name == "ifeval":
        result_text = f"Final result: {round(results['results'][task_name]['inst_level_strict_acc,none'] * 100, 2)}"
        print(result_text)
        output_lines.append(result_text)
    else:
        raise NotImplementedError("Unknown task for final result printing")
    
    os.makedirs("checkpoints", exist_ok=True)
    with open(filename, 'w') as f:
        f.write(f"Strategy: {args.strategy}\n")
        f.write(f"Task: {task_name}\n")
        f.write("=" * 50 + "\n")
        for line in output_lines:
            f.write(line + "\n")
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()