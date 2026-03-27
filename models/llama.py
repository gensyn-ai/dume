import torch
import re
from models.modeling_llama import LlamaConfig, LlamaForCausalLM
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class LlamaWrapper(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        attn_implementation: str = "eager",
        vocab_size=128002,  # matches "meta-llama/Meta-Llama-3-8B" (128000 + 2)
        moe_experts = None,
        k=1,
        temp=0.1,
        _lambda=0.01,
        context_dim=512,
        alpha=None
    ):
        super().__init__()
        config = LlamaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            intermediate_size=4 * hidden_size,
            vocab_size=vocab_size,
            torch_dtype=torch.bfloat16,
        )
        config._attn_implementation = attn_implementation
        self._model = LlamaForCausalLM(config, moe_experts=moe_experts, k=k, temp=temp, _lambda=_lambda, context_dim=context_dim, alpha=alpha).to(dtype=torch.bfloat16)  # type: ignore

    def forward(self, input_ids: torch.Tensor, expert_id=None):
        outputs = self._model(input_ids=input_ids, expert_id=expert_id)
        if isinstance(outputs, tuple):
            all_p = outputs[1]
            all_u = outputs[2]
            outputs = outputs[0]
            return outputs.logits, all_p, all_u
        return outputs.logits

    def set_rr_gate_mode(self, mode: bool):
        self._model.set_rr_gate_mode(mode)

    def generate(self, input_ids, expert_id=None, **kwargs):
        if expert_id is not None:
            original_forward = self._model.forward  # Store original forward method
            
            # Create a wrapper that injects expert_id
            def forward_with_expert_id(*args, **forward_kwargs):
                forward_kwargs['expert_id'] = expert_id
                return original_forward(*args, **forward_kwargs)
            
            self._model.forward = forward_with_expert_id  # Temporarily replace the forward method
            
            try:
                result = self._model.generate(input_ids=input_ids, **kwargs)  # Call generate with modified forward
            finally:
                self._model.forward = original_forward  # Restore original forward method
                
            return result
        else:
            return self._model.generate(input_ids=input_ids, **kwargs)


# additional parameter of intermediate_size
# That allows keeping the same hidden_size and manipulating the intermediate_size for different model sizes
class LlamaWrapperWithMLPSize(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_hidden_layers: int,
        attn_implementation: str = "eager",
        config=None,
        vocab_size=128002,  # matches "meta-llama/Meta-Llama-3-8B" (128000 + 2)
        moe_experts = None,
        k=1,
        temp=0.1,
        _lambda=0.01,
        context_dim=512,
        alpha=None
    ):
        super().__init__()
        if config is None:
            config = LlamaConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                torch_dtype=torch.bfloat16,
            )
            config._attn_implementation = attn_implementation
        
        self._model = LlamaForCausalLM(config, moe_experts=moe_experts, k=k, temp=temp, _lambda=_lambda, context_dim=context_dim, alpha=alpha).to(dtype=torch.bfloat16)  # type: ignore

    def forward(self, input_ids: torch.Tensor, expert_id=None):
        outputs = self._model(input_ids=input_ids, expert_id=expert_id)
        if isinstance(outputs, tuple):
            all_p = outputs[1]
            all_u = outputs[2]
            outputs = outputs[0]
            return outputs.logits, all_p, all_u
        return outputs.logits

    def set_rr_gate_mode(self, mode: bool):
        self._model.set_rr_gate_mode(mode)
    
    def generate(self, input_ids, expert_id=None, **kwargs):
        if expert_id is not None:
            original_forward = self._model.forward  # Store original forward method
            
            # Create a wrapper that injects expert_id
            def forward_with_expert_id(*args, **forward_kwargs):
                forward_kwargs['expert_id'] = expert_id
                return original_forward(*args, **forward_kwargs)
            
            self._model.forward = forward_with_expert_id  # Temporarily replace the forward method
            
            result = self._model.generate(input_ids=input_ids, **kwargs)  # Call generate with modified forward
            self._model.forward = original_forward  # Restore original forward method
                
            return result
        else:
            return self._model.generate(input_ids=input_ids, **kwargs)
    

@register_model("llama_wrapper_eval")
class LLamaWrapperEval(LM):
    
    def __init__(self, model, tokenizer, device="cuda", expert_id=None):
        super().__init__()
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.expert_id = expert_id

    @torch.no_grad()
    def loglikelihood(self, requests):

        input_ids_list = []
        cont_lengths = []

        # 1) Tokenize and build sequences
        print("Tokenizing inputs...")
        for instance in tqdm(requests):

            args = instance.args
            context, continuation = args
            
            # Encode jointly to preserve boundary, then split
            whole_ids = self.tokenizer(context + continuation, return_tensors="pt").input_ids[0].to(self.device)
            context_ids = self.tokenizer(context, return_tensors="pt").input_ids[0].to(self.device)
            continuation_ids = whole_ids[len(context_ids) :]

            input_ids_list.append(torch.cat([context_ids, continuation_ids], dim=0))
            cont_lengths.append(int(continuation_ids.shape[0]))

        # 2) Process in mini-batches to limit memory
        batch_size = 4
        results = []

        print("Computing log-likelihoods...")
        for start in tqdm(range(0, len(input_ids_list), batch_size)):
            end = start + batch_size
            batch_inputs = input_ids_list[start:end]
            batch_cont_lengths = cont_lengths[start:end]

            # Pad this batch
            padded_input_ids = pad_sequence(batch_inputs, batch_first=True, padding_value=self.tokenizer.eos_token_id)

            # Forward + log-probs for this batch
            logits = self.model(padded_input_ids, expert_id=self.expert_id)
            log_probs = torch.log_softmax(logits, dim=-1)

            # Sum continuation token log-probs for this batch
            for i, seq in enumerate(batch_inputs):
                ctx_len = len(seq) - batch_cont_lengths[i]
                total = 0.0
                for j in range(batch_cont_lengths[i]):
                    token_id = seq[ctx_len + j]
                    pos = ctx_len + j - 1
                    total += log_probs[i, pos, token_id].item()
                results.append((float(total), True))

        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests):  # NOT USED
        results = []
        return results

    def generate_until(self, requests):
        # requests: list of prompt strings
        # return list of generated continuations (strings)
        generated = []
        print("Generating outputs...")
        for prompt in tqdm(requests):
            if prompt.task_name == 'gsm8k':
                prompt_text = prompt.doc['question']
            elif 'hendrycks_math' in prompt.task_name:
                prompt_text = prompt.doc['problem']
            elif prompt.task_name == 'ifeval' or prompt.task_name == 'humaneval':
                prompt_text = prompt.doc['prompt']
            else:
                raise NotImplementedError("Unknown task for generation")

            input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.device)
            attention_mask = (input_ids != self.tokenizer.eos_token_id).long()  # 1 for real tokens, 0 for padding
            output_ids = self.model.generate(
                input_ids,
                expert_id=self.expert_id,  # Use stored expert_id
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            generated.append(text)
        return generated