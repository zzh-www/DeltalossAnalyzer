import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import functools

class DeltaLossAnalyzer:
    def __init__(self, model_id, device="cuda", candidates=None):
        self.model_id = model_id
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.hooks = []
        self.layer_inputs = {}
        self.layer_grads = {}
        self.candidates = candidates if candidates else [2, 3, 4, 8]

    def get_calib_dataset(self, n_samples=16, seq_len=512):
        print("Loading calibration dataset...")
        try:
            data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(data["text"])
        except Exception as e:
            print(f"Failed to load wikitext: {e}. Using dummy data.")
            text = "This is a test sentence. " * 1000

        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids

        samples = []
        for _ in range(n_samples):
            i = np.random.randint(0, input_ids.shape[1] - seq_len)
            sample = input_ids[:, i : i + seq_len].to(self.device)
            samples.append(sample)
        return samples

    def register_hooks(self):
        print("Registering hooks...")
        self.remove_hooks()

        def forward_hook(module, input, output, name):
            self.layer_inputs[name] = input[0].detach()
            self.layer_outputs[name] = output.detach() # A_fp

        def backward_hook(module, grad_input, grad_output, name):
            self.layer_grads[name] = grad_output[0].detach() # dL/dA

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "lm_head" not in name:
                self.hooks.append(
                    module.register_forward_hook(
                        functools.partial(forward_hook, name=name)
                    )
                )
                self.hooks.append(
                    module.register_full_backward_hook(
                        functools.partial(backward_hook, name=name)
                    )
                )

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.layer_inputs = {}
        self.layer_outputs = {}
        self.layer_grads = {}

    def fake_quantize(self, weight, bits):
        if bits >= 16:
            return weight

        # Reshape for per-channel quantization (axis=0)
        # Range: [-max_abs, max_abs] (symmetric)
        max_val = weight.abs().amax(dim=1, keepdim=True)
        max_val = torch.clamp(max_val, min=1e-5)

        q_min = -(2 ** (bits - 1))
        q_max = (2 ** (bits - 1)) - 1
        scale = max_val / q_max

        # Quantize and Dequantize
        weight_q = torch.round(weight / scale)
        weight_q = torch.clamp(weight_q, q_min, q_max)
        return weight_q * scale

    def compute_sensitivity(self, samples):
        print("Computing sensitivity...")
        self.register_hooks()
        self.model.eval()
        
        sensitivity_map = {}
        total_tokens = 0

        pbar = tqdm(samples, desc="Calibration")
        for i, sample in enumerate(pbar):
            self.model.zero_grad()
            
            # Forward & Backward Pass
            outputs = self.model(sample, labels=sample)
            loss = outputs.loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            loss.backward()

            # Compute DeltaLoss
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear) and name in self.layer_inputs:
                    if name not in sensitivity_map:
                        sensitivity_map[name] = {b: 0.0 for b in self.candidates}
                    
                    grad_A = self.layer_grads[name]      # [Batch, Seq, Out]
                    input_val = self.layer_inputs[name]   # [Batch, Seq, In]
                    A_fp = self.layer_outputs[name]      # [Batch, Seq, Out]
                    W = module.weight
                    bias = module.bias

                    grad_A_flat = grad_A.reshape(-1, grad_A.shape[-1])
                    input_val_flat = input_val.reshape(-1, input_val.shape[-1])
                    A_fp_flat = A_fp.reshape(-1, A_fp.shape[-1])

                    # Current Batch Token Count
                    n_tokens = grad_A_flat.shape[0]
                    if name == list(sensitivity_map.keys())[0]: # Only count tokens once per sample
                        total_tokens += n_tokens

                    for bits in self.candidates:
                        W_q = self.fake_quantize(W, bits)
                        A_q_flat = torch.matmul(input_val_flat, W_q.t())
                        if bias is not None:
                            A_q_flat += bias
                        
                        # DeltaLoss for this sample: sum of |grad * error|
                        sample_score = (grad_A_flat * (A_fp_flat - A_q_flat)).abs().sum().item()
                        sensitivity_map[name][bits] += sample_score
            
            # Clear buffers
            self.layer_inputs = {}
            self.layer_outputs = {}
            self.layer_grads = {}
            
        # 6. Global average over all tokens in the calibration set
        # This gives the "Expected Loss Increase per Token"
        if total_tokens > 0:
            for name in sensitivity_map:
                for bits in self.candidates:
                    sensitivity_map[name][bits] /= total_tokens

        self.remove_hooks()
        return sensitivity_map

    def allocate_bits(self, sensitivity_map, target_avg_bits):
        print(f"Allocating bits for target avg: {target_avg_bits}...")
        
        layers = sorted(list(sensitivity_map.keys()))
        n_layers = len(layers)
        budget_limit = int(target_avg_bits * n_layers)
        
        # DP Initialization
        # dp[i][j] stores the min loss for the first i layers with total bit budget j
        max_possible_budget = n_layers * max(self.candidates)
        dp = np.full((n_layers + 1, max_possible_budget + 1), np.inf)
        dp[0][0] = 0
        
        # choice[i][j] stores the bit-width chosen for layer i to achieve budget j
        choice = np.zeros((n_layers + 1, max_possible_budget + 1), dtype=int)
        
        layer_costs = [sensitivity_map[name] for name in layers]
        min_bits, max_bits = min(self.candidates), max(self.candidates)

        pbar = tqdm(range(1, n_layers + 1), desc="Searching")
        for i in pbar:
            layer_idx = i - 1
            costs = layer_costs[layer_idx]
            
            # Determine valid range of previous budgets to reduce search space
            min_prev_budget = (i - 1) * min_bits
            max_prev_budget = (i - 1) * max_bits

            for b in self.candidates:
                cost = costs[b]
                
                # Check valid previous states
                for prev_j in range(min_prev_budget, max_prev_budget + 1):
                    if dp[i - 1][prev_j] == np.inf:
                        continue
                    
                    new_j = prev_j + b
                    new_loss = dp[i - 1][prev_j] + cost
                    
                    if new_loss < dp[i][new_j]:
                        dp[i][new_j] = new_loss
                        choice[i][new_j] = b
            
            # Logging
            valid_losses = dp[i][dp[i] != np.inf]
            if len(valid_losses) > 0:
                pbar.set_postfix({"min_delta": f"{np.min(valid_losses):.2e}"})

        # Find best configuration within budget limit
        # We search in range [min_possible_budget, target_budget_limit]
        best_loss = np.inf
        best_j = -1
        
        search_start = n_layers * min_bits
        search_end = min(budget_limit, max_possible_budget)

        for j in range(search_start, search_end + 1):
            if dp[n_layers][j] < best_loss:
                best_loss = dp[n_layers][j]
                best_j = j
                
        if best_j == -1:
            print("Could not find a valid configuration within budget!")
            return None
            
        # Reconstruct Allocation
        allocation = {}
        curr_j = best_j
        for i in range(n_layers, 0, -1):
            b = int(choice[i][curr_j])
            allocation[layers[i - 1]] = b
            curr_j -= b
            
        return allocation