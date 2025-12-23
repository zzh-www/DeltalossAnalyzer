import argparse
import json
import os
from delta_loss import DeltaLossAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description="SignRoundV2 Adaptive Bit-Width Analyzer"
    )
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.7B", help="Model ID"
    )
    parser.add_argument(
        "--bits", type=float, default=4.0, help="Target average bit-width"
    )
    parser.add_argument(
        "--samples", type=int, default=16, help="Number of calibration samples"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Directory to save outputs"
    )
    parser.add_argument(
        "--candidates",
        type=int,
        nargs="+",
        default=None,
        help="List of candidate bit-widths (e.g., 2 4 8)",
    )

    args = parser.parse_args()

    # Create a safe model name for file paths
    safe_model_name = args.model.replace("/", "_")
    model_output_dir = os.path.join(args.output_dir, safe_model_name)

    # Ensure output directory exists
    os.makedirs(model_output_dir, exist_ok=True)

    candidates = args.candidates
    if candidates:
        print(f"Using custom candidates: {candidates}")

    print(f"Starting analysis for {args.model}...")
    analyzer = DeltaLossAnalyzer(args.model, candidates=candidates)

    samples = analyzer.get_calib_dataset(n_samples=args.samples)
    sensitivity = analyzer.compute_sensitivity(samples)

    # Save raw sensitivity
    sens_path = os.path.join(model_output_dir, "sensitivity.json")
    with open(sens_path, "w") as f:
        json.dump(sensitivity, f, indent=2)
    print(f"Sensitivity saved to {sens_path}")

    allocation = analyzer.allocate_bits(sensitivity, args.bits)

    if allocation:
        out_path = os.path.join(model_output_dir, "bit_allocation.json")
        print(f"Allocation complete. Saving to {out_path}")
        with open(out_path, "w") as f:
            json.dump(allocation, f, indent=2)

        # Print stats
        avg_bits = sum(allocation.values()) / len(allocation)
        print(f"Final average bits: {avg_bits:.2f}")
    else:
        print("Allocation failed.")


if __name__ == "__main__":
    main()
