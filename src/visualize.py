import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_layer_info(name):
    # Extract block index
    # Pattern: layers.10.xxx or h.10.xxx
    match = re.search(r'\.(\d+)\.', name)
    if match:
        block_idx = int(match.group(1))
    else:
        # Handle cases like lm_head or embed_tokens which don't have a numeric block index
        # We assign them to -1 (start) or 9999 (end) for sorting, but might exclude from main plot
        block_idx = -1 

    # Extract module type
    if "q_proj" in name: module_type = "Q_proj"
    elif "k_proj" in name: module_type = "K_proj"
    elif "v_proj" in name: module_type = "V_proj"
    elif "o_proj" in name: module_type = "O_proj"
    elif "gate_proj" in name: module_type = "Gate_proj"
    elif "up_proj" in name: module_type = "Up_proj"
    elif "down_proj" in name: module_type = "Down_proj"
    elif "lm_head" in name: module_type = "Head"
    else: module_type = "Other"
    
    return block_idx, module_type

def plot_sensitivity_profile(sensitivity, output_dir):
    plt.figure(figsize=(16, 8))
    
    layers = sorted(list(sensitivity.keys()))
    candidates = sorted(list(sensitivity[layers[0]].keys()))
    
    data = []
    for layer in layers:
        block_idx, module_type = get_layer_info(layer)
        if block_idx == -1 and module_type != "Head": continue # Skip embeddings/norm if any
        
        # Place Head at the end
        if module_type == "Head":
             # Find max index to place head after
             max_idx = max([get_layer_info(l)[0] for l in layers])
             block_idx = max_idx + 1

        for bits in candidates:
            b = int(bits)
            score = sensitivity[layer][str(bits)]
            data.append({
                "Block Index": block_idx, 
                "Sensitivity": score, 
                "Bits": b,
                "Module": module_type
            })
            
    df = pd.DataFrame(data)
    
    # Plot: X=Block Index, Y=Sensitivity, Hue=Module, Style=Bits
    sns.lineplot(
        data=df, 
        x="Block Index", 
        y="Sensitivity", 
        hue="Module", 
        style="Bits",
        markers=True,
        dashes=False,
        palette="tab10",
        linewidth=1.5,
        alpha=0.8
    )
    
    plt.title("Layer Sensitivity Profile (DeltaLoss) per Block")
    plt.xlabel("Transformer Block Index")
    plt.ylabel("DeltaLoss per Token")
    plt.yscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "sensitivity_profile.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_allocation_profile(allocation, model_info, output_dir):
    plt.figure(figsize=(16, 8))
    
    layers = sorted(list(allocation.keys()))
    
    # Organize data by block index
    block_data = {} # {block_idx: [{module, params, bits}]}
    
    for layer in layers:
        block_idx, module_type = get_layer_info(layer)
        if block_idx == -1 and module_type != "Head": continue
        
        if module_type == "Head":
             max_idx = max([get_layer_info(l)[0] for l in layers])
             block_idx = max_idx + 1
             
        bits = allocation[layer]
        params = model_info.get(layer, 0)
        
        if block_idx not in block_data:
            block_data[block_idx] = []
        block_data[block_idx].append({
            "module": module_type,
            "params": params,
            "bits": bits,
            "layer_name": layer
        })

    sorted_blocks = sorted(block_data.keys())
    
    # Create colormap for bits
    unique_bits = sorted(list(set(allocation.values())))
    # Use a diverging or sequential palette
    palette = sns.color_palette("viridis", len(unique_bits))
    bit_color_map = {b: palette[i] for i, b in enumerate(unique_bits)}
    
    bottoms = np.zeros(len(sorted_blocks))
    
    # We need to stack bars. 
    # To make it readable, we can fix the order of modules within a stack
    module_order = ["Q_proj", "K_proj", "V_proj", "O_proj", "Gate_proj", "Up_proj", "Down_proj", "Head", "Other"]
    
    # Collect handles for legend
    handles = []
    labels = []
    
    # Iterate through each block and plot segments
    for i, block_idx in enumerate(sorted_blocks):
        segments = block_data[block_idx]
        # Sort segments by standard module order
        segments.sort(key=lambda x: module_order.index(x["module"]) if x["module"] in module_order else 99)
        
        # Calculate total params for this block to determine label threshold
        block_total_params = sum(s["params"] for s in segments)
        
        current_bottom = 0
        for seg in segments:
            color = bit_color_map[seg["bits"]]
            mod = seg["module"]
            
            # Apply hatch for MLP layers to distinguish from Attention
            hatch = '//' if mod in ["Gate_proj", "Up_proj", "Down_proj"] else ''
            
            bar = plt.bar(
                i, 
                seg["params"], 
                bottom=current_bottom, 
                color=color, 
                edgecolor='white', 
                linewidth=0.5,
                width=0.8,
                hatch=hatch
            )
            
            # Add text label if segment is large enough (> 2% of block height)
            if seg["params"] / block_total_params > 0.02:
                # Map module to short abbreviation
                short_map = {
                    "Q_proj": "Q", "K_proj": "K", "V_proj": "V", "O_proj": "O",
                    "Gate_proj": "G", "Up_proj": "U", "Down_proj": "D", "Head": "H"
                }
                label = short_map.get(mod, "")
                
                # Calculate center position
                y_center = current_bottom + seg["params"] / 2
                
                # Choose text color based on background (heuristic: light for dark bars)
                # But simple black/white usually works. Let's use white with shadow or just black.
                # Given viridis, 2-bit is usually purple (dark), 4-bit yellow/green (light).
                # Actually simpler: always white text with a thin outline usually works, or just black.
                # Let's try white text which contrasts well with most saturated colors, 
                # maybe black if it's very light. 
                # For simplicity/robustness, we stick to white, maybe bold.
                plt.text(
                    i, 
                    y_center, 
                    label, 
                    ha='center', 
                    va='center', 
                    color='white', 
                    fontsize=8, 
                    fontweight='bold'
                )

            current_bottom += seg["params"]
    
    # Custom Legend for Bits
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=bit_color_map[b], edgecolor='white', label=f'{b}-bit') for b in unique_bits]
    # Add legend entry for Texture
    legend_elements.append(Patch(facecolor='white', edgecolor='gray', hatch='//', label='MLP (G/U/D)'))
    legend_elements.append(Patch(facecolor='white', edgecolor='gray', label='Attn (Q/K/V/O)'))
    
    plt.legend(handles=legend_elements, title="Legend", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title("Parameter-Weighted Bit Allocation per Block (Stacked Bar)\nLabels: G=Gate, U=Up, D=Down, Q/K/V/O=Attn")
    plt.xlabel("Transformer Block Index")
    plt.ylabel("Parameter Count (per block)")
    
    # Set x-ticks
    plt.xticks(range(len(sorted_blocks)), sorted_blocks)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "allocation_profile.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def plot_bit_distribution(allocation, model_info, output_dir):
    # Calculate weighted distribution
    stats = {}
    total_params = 0
    
    for layer, bits in allocation.items():
        params = model_info.get(layer, 0)
        total_params += params
        stats[bits] = stats.get(bits, 0) + params
        
    # Prepare data for plotting
    labels = [f"{b}-bit" for b in stats.keys()]
    sizes = [stats[b] for b in stats.keys()]
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
    plt.title(f"Parameter Distribution by Bit-Width\n(Total Params: {total_params/1e9:.2f}B)")
    
    save_path = os.path.join(output_dir, "bit_distribution.png")
    plt.savefig(save_path)
    print(f"Saved {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize DeltaLoss Analysis Results")
    parser.add_argument("output_dir", type=str, help="Path to the output directory containing JSON files")
    args = parser.parse_args()
    
    sens_path = os.path.join(args.output_dir, "sensitivity.json")
    alloc_path = os.path.join(args.output_dir, "bit_allocation.json")
    info_path = os.path.join(args.output_dir, "model_info.json")
    
    if not os.path.exists(sens_path) or not os.path.exists(alloc_path):
        print(f"Error: Missing json files in {args.output_dir}")
        return

    print(f"Generating visualizations for {args.output_dir}...")
    
    sensitivity = load_json(sens_path)
    allocation = load_json(alloc_path)
    
    # Load model info
    if os.path.exists(info_path):
        model_info = load_json(info_path)
    else:
        print("Warning: model_info.json not found. Creating dummy params for visualization.")
        model_info = {k: 1 for k in allocation.keys()}

    # Generate Sensitivity Plot
    plot_sensitivity_profile(sensitivity, args.output_dir)
    
    # Generate Allocation Plot (Stacked Bar)
    plot_allocation_profile(allocation, model_info, args.output_dir)
    
    # Generate Distribution Plot
    plot_bit_distribution(allocation, model_info, args.output_dir)

if __name__ == "__main__":
    main()
