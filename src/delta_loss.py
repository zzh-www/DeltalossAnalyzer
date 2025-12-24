import torch
import functools
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

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
        self.layer_outputs = {}
        self.layer_grads = {}
        self.layer_params = {} # Store parameter counts
        self.candidates = candidates if candidates else [2, 3, 4, 8]

    def get_calib_dataset(self, n_samples=16, seq_len=1024):
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
                        self.layer_params[name] = module.weight.shape[0] *  module.weight.shape[1] # Record params
                    
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
        """
        Performs bit allocation using Lagrange Multipliers to handle different layer sizes.
        Objective: Minimize sum(S_l) subject to sum(P_l * b_l) / sum(P_l) <= target_avg_bits
        """
        print(f"Allocating bits (Weighted) for target avg: {target_avg_bits}...")
        
        layers = sorted(list(sensitivity_map.keys()))
        diff_params = list(set(self.layer_params.values()))
        scale = functools.reduce(torch.gcd, torch.tensor(diff_params, dtype=torch.long)).item()
        for name in layers:
            self.layer_params[name] /= scale
        total_params = sum(self.layer_params.values())
        target_total_bits = target_avg_bits * total_params
        
        # We search for a lambda such that choosing b_l to minimize (S_l + lambda * P_l * b_l)
        # meets the budget constraint.
        
        low_lambda = 0.0
        high_lambda = max(list(set(self.layer_params.values()))) # Start with a large range
        best_allocation = None
        
        # Binary search for lambda
        for step in range(128):
            mid_lambda = (low_lambda + high_lambda) / 2
            current_allocation = {}
            current_total_bits = 0
            
            for name in layers:
                params = self.layer_params[name]
                costs = sensitivity_map[name]
                
                # Pick bits that minimize: Sensitivity + lambda * Total_Bits_In_Layer
                # b = argmin ( costs[b] + mid_lambda * params * b )
                best_b = min(self.candidates, key=lambda b: costs[b] + mid_lambda * params * b)
                
                current_allocation[name] = best_b
                current_total_bits += params * best_b
            
            if current_total_bits <= target_total_bits:
                best_allocation = current_allocation
                actual_avg = sum(best_allocation[n] * self.layer_params[n] for n in layers) / total_params
                actual_loss = sum(sensitivity_map[n][best_allocation[n]] for n in layers)
                print(f"step {step}: avg_bit {actual_avg}, loss: {actual_loss:.5f}")
                high_lambda = mid_lambda # Try to lower lambda to get closer to budget
                if abs(current_total_bits - target_total_bits) / target_total_bits < 0.0001:
                    break
            else:
                low_lambda = mid_lambda # Need more penalty for bits, increase lambda

        if best_allocation:
            actual_avg = sum(best_allocation[n] * self.layer_params[n] for n in layers) / total_params
            actual_loss = sum(sensitivity_map[n][best_allocation[n]] for n in layers)
            print(f"Allocation found. Actual Weighted Avg Bits: {actual_avg:.4f}, loss: {actual_loss:.5f}")
            
        return best_allocation