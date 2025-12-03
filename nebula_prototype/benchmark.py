import torch
import time
from model.nebula import NebulaModel
from model.config import NEBULA_CONFIGS

def benchmark():
    config_name = "500M"
    batch_size = 16
    seq_len = 1024
    num_steps = 100
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}")
    
    # Load Model
    config = NEBULA_CONFIGS[config_name]
    model = NebulaModel(config).to(device)
    
    # Check Mamba
    try:
        import mamba_ssm
        print("Mamba-SSM: INSTALLED")
    except ImportError:
        print("Mamba-SSM: MISSING (Using Mock)")

    # Compile
    print("Compiling...")
    model = torch.compile(model)
    
    # Synthetic Data
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    print("Warmup...")
    for _ in range(10):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(input_ids)
            
    torch.cuda.synchronize()
    
    # Benchmark
    print("Running benchmark...")
    t0 = time.time()
    total_tokens = 0
    
    for _ in range(num_steps):
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            _ = model(input_ids)
        total_tokens += batch_size * seq_len
        
    torch.cuda.synchronize()
    t1 = time.time()
    
    dt = t1 - t0
    tps = total_tokens / dt
    
    print(f"Throughput: {tps:,.0f} tokens/sec")
    print(f"Total time: {dt:.2f}s for {total_tokens:,} tokens")

if __name__ == "__main__":
    benchmark()
