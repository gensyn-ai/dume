import argparse


MODEL_MAP = {
    "math": "MergeBench/Llama-3.2-3B_math",
    "multilingual": "MergeBench/Llama-3.2-3B_multilingual",
    "coding": "MergeBench/Llama-3.2-3B_coding",
    "instruction": "MergeBench/Llama-3.2-3B_instruction"
}

STRATEGY_CHOICES = [
    "math_expert",
    "multilingual_expert",
    "coding_expert",
    "instruction_expert",
    "model_averaging",
    "oracle",
    "random_routing",
    "DUME",
    "BTX",
    "DUMEplus"
]

TASK_MAP = {
    "math": "gsm8k",
    # "multilingual": "m_mmlu",
    "multilingual": "m_arc",
    "coding": "humaneval",
    "instruction": "ifeval"
}

TASK = 'instruction'
STRATEGY = 'math_expert'


def get_args():
    
    parser = argparse.ArgumentParser(description="Evaluate MergeBench models on various tasks")
    
    parser.add_argument("--task", type=str, choices=list(TASK_MAP.keys()), default=TASK, help="Task to evaluate on")
    parser.add_argument("--strategy", type=str, choices=STRATEGY_CHOICES, default=STRATEGY, help="Which experiment to run")
    parser.add_argument("--_lambda", type=float, default=1e-2, help="Tikhonov regularization parameter")
    parser.add_argument("--k", type=int, default=1, help="K of top-K routing")
    parser.add_argument("--temp", type=float, default=1e-1, help="Softmax temperature for routing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--domains", type=str, nargs='+', default=list(TASK_MAP.keys()), help="Domains for average strategy")
    parser.add_argument("--use_test_distribution", type=bool, default=True, help="Whether to use test distribution for DUME statistics collection")
    parser.add_argument("--num_parameters", type=float, default=3e9, help="Number of parameters for the base dense model experts")
    parser.add_argument('--num_dume_tokens', type=int, default=1024, help='Number of tokens to use for DUME statistics collection')
    parser.add_argument('--max_num_samples', type=int, default=820)
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for BTX')
    parser.add_argument('--num_btx_steps', type=int, default=1000, help='Number of BTX training steps')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha parameter for BTX (if None, no distillation loss is used)')
    args = parser.parse_args()

    task_name = TASK_MAP[args.task]
    if 'expert' in args.strategy:
        model_name = MODEL_MAP[args.strategy.split('_')[0]]
    elif args.strategy in ("model_averaging", "oracle", "random_routing", "DUME", "BTX", "DUMEplus"):
        model_name = list(MODEL_MAP.values())
    else:
        raise NotImplementedError(f"Strategy {args.strategy} not implemented")

    return args, model_name, task_name
