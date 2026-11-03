# Training-Free Dynamic Upcycling of Expert Language Models

This repository contains the code and instructions to replicate the experiments in the paper [Training-free Dynamic Upcycling of Expert Language Models](https://arxiv.org/pdf/2603.29765), accepted at **NeurIPS 2026**.

![DUME](./assets/method.png)

Training LLM models is prohibitively expensive, and they often lack domain-specific expertise because they rely on general knowledge datasets. Expertise finetuning can address this issue; however, it often leads to overspecialization, and developing a single multi-domain expert remains difficult due to diverging objectives. Furthermore, multitask training is challenging due to interference and catastrophic forgetting. Existing work combines dense experts within a Mixture of Experts (MoE) architecture, but still requires multitask finetuning. To address these issues, we introduce **Dynamic Upcycling MoE** (**DUME**), a novel approach that reuses dense experts trained on different domains to construct a unified MoE model. Our method builds a single multitask model that preserves the capabilities of the original dense experts without requiring additional training. DUME is both cost-efficient and scalable: by leveraging the closed-form solution of ridge regression, it eliminates the need for further optimization and enables experts to be added or removed dynamically. Across causal language modeling and reasoning tasks, with models from 115M to 9B parameters, DUME consistently outperforms the baselines; in the R LARGE reasoning setting (Gemma2-9B), it reaches 104.0% of the dense experts' performance on average.

## Install the environment

From the root directory of this repository, run the following commands:

```
conda env create -f env.yml
curl -fsSL https://deno.land/install.sh | sh
```

## Datasets setup

See our datasets setup guide [here](dataset/README.md).


## Reasoning scripts

We provide two reasoning settings, both using [MergeBench](https://huggingface.co/MergeBench) experts:

| Setting | Model | Domains | Flag |
|---|---|---|---|
| R MEDIUM | Llama3-3B | Mathematics (GSM8K), Multilingual (M_ARC), Coding (HumanEval), Instruction following (IFEval) | `--model-type llama` (default) |
| R LARGE | Gemma2-9B | Same as R MEDIUM | `--model-type gemma` |

- Generate and save DUME MoErged model on the four domains: ```python reasoning_generate_dume_params.py --model-type [llama|gemma]```
- Evaluate dense experts, merged or MoErged models with ten different strategies (```math_expert, multilingual_expert, coding_expert, instruction_expert, model_averaging, oracle, random_routing, DUME, BTX, DUMEplus```) on one of the four domains (```math, multilingual, coding, instruction```): ```python reasoning_eval.py --model-type [llama|gemma] --strategy [strategy] --task [task]```

## Causal language modeling scripts

We provide two causal language modeling settings:

| Setting | Model | Domains | Flag |
|---|---|---|---|
| CLM SMALL | Llama3, 115M | Coding, Mathematics, Physics, History and events, Philosophy and thinking | `--setting clm` / `--scenario clm` |
| CLM MEDIUM | Llama3, 1B | Mathematics, Biology, Legal, Dialogue | `--setting clm2` / `--scenario clm2` |

- Pretrain the Llama model on the OpenWebText Corpus dataset: ```clm_pretrain.py``` (add ```--num-parameters 1e9``` to pretrain the CLM MEDIUM model)
- Train the Llama experts on the CLM SMALL domains: ```clm_train_experts.py```
- Train the Llama experts on the CLM MEDIUM domains: ```clm2_train_experts.py```
- Evaluate each Llama expert on all the CLM SMALL domains: ```clm_experts_eval.py```
- Evaluate each Llama expert on all the CLM MEDIUM domains: ```clm2_experts_eval.py```
- Evaluate the Llama merged model with six different merging strategies (```average, task-arithmetic, ties-merging, DARE, Fisher, consensus```) across all the domains of a setting: ```python clm_weights_avg_eval.py --setting [clm|clm2] --merging-strategy [strategy]```
- Evaluate the Llama MoErged model with six different strategies (```oracle, random_routing, DUME, DUMErec, BTX, DUMEplus```): ```python clm_moerging_strategies_eval.py --setting [clm|clm2] --method [method_name]```
  - ```DUMErec``` computes the DUME routers block by block, using the routers already computed for the previous blocks when collecting the statistics for the next one.
  - ```--merging-strategy``` selects how the non-MLP layers of the experts are merged (same options as above).
- Evaluate the Llama MoErged model when adding or removing experts incrementally: ```python clm_incremental_domains.py --scenario [clm|clm2] --method [DUME|BTX] --setting [add|remove]```
  - ```--domains_order``` sets the order in which experts are added or removed. It defaults to the CLM SMALL domains, so pass ```--domains_order math bio legal dialogue``` (or another order) with ```--scenario clm2```.
  - The incremental strategies in the paper (Figure 3) map to: *initial expert params* (default), *frozen params* (```--freeze_attn```), *moving avg params* (```--update_model```), *all averaged params* (```--average_experts```).

The scripts default to different settings: ```clm_weights_avg_eval.py``` and ```clm_moerging_strategies_eval.py``` default to CLM MEDIUM, while ```clm_incremental_domains.py``` defaults to CLM SMALL. When running a script on the other setting, also pass the matching model size: ```1.15e8``` for CLM SMALL or ```1e9``` for CLM MEDIUM, with ```--num-parameters``` (```--num_parameters``` in ```clm_moerging_strategies_eval.py``` and ```clm_incremental_domains.py```).

## Citation

```
@inproceedings{fani2026training,
  title={Training-Free Dynamic Upcycling of Expert Language Models},
  author={Fan{\`i}, Eros and Ersoy, O{\u{g}}uzhan},
  booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
  year={2026},
}
```
