import argparse
import os
import torch
from safetensors.torch import load_file as safe_load_file
from tqdm import tqdm


# We implemented code from the implementation of the paper "PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning". Their github repository is:  https://github.com/Vincent-HKUSTGZ/PEFTGuard for detail implementation of PeftGuard


def convert_safetensor_to_pth(src_path: str,
                               dst_path: str,
                               gpu_id: int = 0,
                               num_layers: int = 32) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for GPU‑side tensor ops.")

    torch.cuda.set_device(gpu_id)

    # ↳ load all LoRA adapter weights onto the specified GPU
    adapters_weights = safe_load_file(src_path, device=f"cuda:{gpu_id}")

    # Dynamically infer dimensions from the first layer's q_proj
    lora_A_sample = adapters_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight']
    lora_B_sample = adapters_weights['base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight']
    hidden_size, rank = lora_B_sample.shape
    _, in_features = lora_A_sample.shape

    # Pre‑allocate container (kept on GPU for speed, moved to CPU just before save)
    results = torch.empty((hidden_size, in_features, num_layers * 2), dtype=torch.float32, device=f"cuda:{gpu_id}")

    for layer in tqdm(range(num_layers), desc="Converting layers"):
        # ---- Q ----
        lora_A_q = adapters_weights[f'base_model.model.model.layers.{layer}.self_attn.q_proj.lora_A.weight']
        lora_B_q = adapters_weights[f'base_model.model.model.layers.{layer}.self_attn.q_proj.lora_B.weight']
        results[:, :, layer * 2] = torch.matmul(lora_B_q, lora_A_q)

        # ---- V ----
        lora_A_v = adapters_weights[f'base_model.model.model.layers.{layer}.self_attn.v_proj.lora_A.weight']
        lora_B_v = adapters_weights[f'base_model.model.model.layers.{layer}.self_attn.v_proj.lora_B.weight']
        results[:, :, layer * 2 + 1] = torch.matmul(lora_B_v, lora_A_v)

    # Move to CPU before serialisation to minimise GPU memory spikes
    torch.save(results.cpu(), dst_path)

    # Clean‑up
    del adapters_weights, results
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA adapter safetensors to consolidated .pth format")
    parser.add_argument("--src", help="Path to adapter_model.safetensors")
    parser.add_argument("--dst", help="Destination .pth file path")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device ID (default: 0)")
    parser.add_argument("--layers", type=int, default=32, help="Number of transformer layers (default: 32)")
    args = parser.parse_args()

    convert_safetensor_to_pth(args.src, args.dst, gpu_id=args.gpu, num_layers=args.layers)
