# Training-Free Dynamic Upcycling of Expert Language Models

This repository contains the code and instructions to replicate the experiments in the paper [Training-free Dynamic Upcycling of Expert Language Models](https://arxiv.org/pdf/2603.29765).

![DUME](./assets/method.png)

Training LLM models is prohibitively expensive, and they often lack domain-specific expertise because they rely on general knowledge datasets. Expertise finetuning can address this issue; however, it often leads to overspecialization, and developing a single multi-domain expert remains difficult due to diverging objectives. Furthermore, multitask training is challenging due to interference and catastrophic forgetting. To address these issues, we introduce **Dynamic Upcycling MoE** (**DUME**), a novel approach that reuses dense experts trained on different domains to construct a unified Mixture of Experts (MoE) model. Our method builds a single multitask model that preserves the capabilities of the original dense experts without requiring additional training. DUME is both cost-efficient and scalable: by leveraging the closed-form solution of ridge regression, it eliminates the need for further optimization and enables experts to be added dynamically while maintaining the model’s original performance. 

## Install the environment

From the root directory of this repository, run the following commands:

```
conda env create -f env.yml
curl -fsSL https://deno.land/install.sh | sh
```

## Datasets setup

See our datasets setup guide [here](dataset/README.md).


## Causal language modeling scripts

- Pretrain the Llama model on the OpenWebText Corpus dataset: ```clm_pretrain.py```
- Train the Llama experts on the five clm domains: ```clm_train_experts.py```
- Evaluate each Llama expert on all the five domains: ```clm_experts_eval.py```
- Evaluate the Llama merged model with weights averaging across all the five clm domains: ```clm_weights_avg_eval.py```
- Evaluate the Llama MoErged model with five different strategies (```oracle, random_routing, DUME, BTX, DUMEplus```): ```python clm_moerging_strategies_eval.py --method [method_name]```

## Reasoning scripts

- Generate and save DUME MoErged model on the four domains: ```reasoning_generate_dume_params.py```
- Evaluate Llama dense experts, merged or MoErged models with nine different strategies (```math_expert, multilingual_expert, coding_expert, instruction_expert, model_averaging, oracle, random_routing, DUME, BTX, DUMEplus```) on one of the four domains (```math, multilingual, coding, instruction```): ```python reasoning_generate_dume_params.py --strategy [strategy] --task [task]```

## Citation

```
@misc{fani2026training,
  title={Training-Free Dynamic Upcycling of Expert Language Models},
  author={Fan{\`i}, Eros and Ersoy, O{\u{g}}uzhan},
  booktitle={1st Workshop on Scaling Post-training for LLMs (SPOT) @ ICLR 2026},
  year={2026},
}
```
