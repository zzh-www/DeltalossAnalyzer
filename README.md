# DeltaLoss Analyzer for Qwen/Qwen3-1.7B

This tool implements the **Adaptive Bit-Width** algorithm described in SignRoundV2, using the DeltaLoss sensitivity metric to allocate optimal bit-widths for each layer of an LLM.

## Setup

1.  **Install Dependencies:**
    ```bash
    uv sync
    # Or manually:
    pip install torch transformers accelerate datasets sentencepiece protobuf
    ```

## Usage

To run the analysis on `Qwen/Qwen3-1.7B` (or any other model):

```bash
    uv run src/main.py --model Qwen/Qwen3-1.7B --bits 4.0 --samples 128
    ```

### Arguments:
*   `--model`: Model ID from HuggingFace (default: `Qwen/Qwen2.5-1.5B`).
*   `--bits`: Target average bit-width (e.g., 2.0, 4.0).
*   `--samples`: Number of calibration samples (default: 16). Higher is more accurate but slower.
*   `--output`: Output JSON file for the bit allocation map (default: `bit_allocation.json`).

## Algorithm Details

1.  **Calibration**: Loads a subset of Wikitext2.
2.  **Sensitivity Analysis**:
    *   Computes gradients of the loss w.r.t. output activations ($\nabla_{A_i} \mathcal{L}$). 
    *   Simulates quantization for candidates {2, 3, 4, 8} bits.
    *   Calculates DeltaLoss: $\mathcal{S}_i = \sum |\nabla_{A_i} \mathcal{L} \odot (A_i - \hat{A}_i)|$.
3.  **Bit Allocation**:
    *   Uses Dynamic Programming to minimize total DeltaLoss subject to the bit budget constraint.

## Output

The script produces a JSON file mapping each linear layer name to its assigned bit-width.

```