import os
import torch
import logging
import typing as T
import jaxtyping as Jt
import torch._inductor.config
import torch._logging as tlog

from torch.nn.attention import flex_attention

tlog.set_logs(inductor=logging.ERROR, perf_hints=False)
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
torch._inductor.config.verbose_progress = False
torch._inductor.config.debug = False


T_Backend = T.Literal["torch-math", "torch-flash", "torch-cudnn", "torch-flex", "torch-memeff"]

# NOTE: This line is very important - it enables the Triton compile for flex attn.
#       Without it you may observe a 50x slow down.
os.environ['TRITON_DYNAMIC_SHAPES'] = '1'
fn_flex_attention = torch.compile(
    flex_attention.flex_attention,
    mode='reduce-overhead',
    dynamic=True
)


def __torch_flex_scoremod(attention_bias: torch.Tensor):
    attention_bias = attention_bias.squeeze(1, 2)
    def impl(score: torch.Tensor, batch: torch.Tensor, head: torch.Tensor, q_idx: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
        return score + attention_bias[batch, k_idx]
    return impl


def sdpa_dispatch(
    q: Jt.Float[torch.Tensor, "B H N C"],
    k: Jt.Float[torch.Tensor, "B H N C"],
    v: Jt.Float[torch.Tensor, "B H N C"],
    attn_bias: Jt.Float[torch.Tensor, "B 1 1 N"] | None,
    backend: T_Backend
) -> torch.Tensor:
    """
    There used to have N sdpa implementations...
    
    'why there are so many implementations for (not exactly) the same thing? let's create one to rule them all!'
    
    There used to have N+1 sdpa implementations...
    """
    B, H, N, C = q.shape
    
    match backend:
        case "xformer-memeff":
            raise NotImplementedError()
        
        case "torch-cudnn"   :
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        
        case "torch-flash"   :
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        
        case "torch-math"    :
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        
        case "torch-flex"    :
            if attn_bias is None:
                return fn_flex_attention(q, k, v)
            else:
                score_mod = __torch_flex_scoremod(attn_bias)
                return fn_flex_attention(q, k, v, score_mod=score_mod)
        
        case "torch-memeff"  :
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        
        case "tridao-flash"  :
            raise NotImplementedError()
        
        case _:
            raise Exception(f"Unsupported backend {backend}")



def benchmark_sdpa(B: int, H: int, N: int, C: int, attn_bias: bool, backend: T_Backend, warmup: int=5):
    import torch.utils.benchmark as benchmark

    # Data Prep
    q = torch.randn(B, H, N, C, device="cuda", dtype=torch.float16, requires_grad=False)
    k = torch.randn(B, H, N, C, device="cuda", dtype=torch.float16, requires_grad=False)
    v = torch.randn(B, H, N, C, device="cuda", dtype=torch.float16, requires_grad=False)
    attn_bias_tensor = torch.randn(B, 1, 1, N, device="cuda", dtype=torch.float16) if attn_bias else None
    # End

    # Correctness check
    try:
        backend_output = sdpa_dispatch(q, k, v, attn_bias_tensor, backend)
    except:
        print(f"SDPA Backend = {backend} does not support current setup. Skipped.")
        return 

    referen_output = sdpa_dispatch(q, k, v, attn_bias_tensor, "torch-math")
    assert torch.allclose(backend_output, referen_output, rtol=1e-3, atol=1e-3)
    # End
    
    # Warmup
    for _ in range(warmup):
        sdpa_dispatch(q, k, v, attn_bias_tensor, backend)
    # End

    # Benchmarking
    stmt = "sdpa_dispatch(q, k, v, attn_bias_tensor, backend)"
    globals_dict = {
        "sdpa_dispatch": sdpa_dispatch,
        "q": q,
        "k": k,
        "v": v,
        "attn_bias_tensor": attn_bias_tensor,
        "backend": backend,
    }
    timer = benchmark.Timer(
        label=f"SDPA Backend = {backend}",
        sub_label=f"{B=} {H=} {C=} {N=}, {attn_bias=}, dtype=fp16",
        stmt=stmt,
        globals=globals_dict,
        num_threads=torch.get_num_threads(),
    )
    print(timer.timeit(100))
    # End


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    setups = [
        dict(B=8, H=16, N=2048    , C=64, attn_bias=True) ,  # VGGT Frame Attention
        dict(B=1, H=16, N=8 * 2048, C=64, attn_bias=True) ,  # VGGT Global Attention
        dict(B=8, H=16, N=2048    , C=64, attn_bias=False),  # VGGT Frame Attention
        dict(B=1, H=16, N=8 * 2048, C=64, attn_bias=False),  # VGGT Global Attention
    ]
    
    for setup in setups:
        for backend in ["xformer-memeff", "torch-math", "torch-flash", "torch-cudnn", "torch-flex", "torch-memeff", "tridao-flash"]:
            benchmark_sdpa(backend=backend, **setup)
        print("\n\n\n")
