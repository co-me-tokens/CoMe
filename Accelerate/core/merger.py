import math
import torch
import jaxtyping as Jt


# NOTE: This requires additional compile.
from CUExt.CUDAExtension import ext as Extension


class TokenMerger:
    def __init__(self, start_idx: int, grp_size: int):
        self.start_idx = start_idx
        self.grp_size  = grp_size
    
    @torch.no_grad()
    def merge(self, token: Jt.Float[torch.Tensor, "B N*G C"], dst_lidx: Jt.UInt64[torch.Tensor, "B N*G"]):
        with torch.cuda.nvtx.range("cuda_tm/merge", color='yellow'):
            B, N, C = token.shape
            patch_token = token[:, self.start_idx:].reshape(B, (N-self.start_idx)//self.grp_size, self.grp_size, C)
            merge_token = Extension.token_grpcontract_v2_merge(self.start_idx, patch_token, dst_lidx, None)
            merge_token[:, :self.start_idx] = token[:, :self.start_idx]
            return merge_token, (B, N, C)
    
    @torch.no_grad()
    def split(self, token: Jt.Float[torch.Tensor, "B N2 C"], tok_shape: tuple[int, int, int], dst_lidx: Jt.UInt64[torch.Tensor, "B N*G"]):
        with torch.cuda.nvtx.range("cuda_tm/split", color='yellow'):
            result = torch.empty(tok_shape, device=token.device, dtype=token.dtype)
            result[:, :self.start_idx] = token[:, :self.start_idx]
            result[:, self.start_idx:] = torch.take_along_dim(token, dst_lidx.unsqueeze(-1), dim=1)
            return result
    
    @torch.no_grad()
    @staticmethod
    def align_boolean_mask(mask: Jt.Bool[torch.Tensor, "B N//G"]) -> Jt.Bool[torch.Tensor, "B N//G"]:
        """
        Align the number of 'True' value in each row of the mask tensor to minimum number of 'True' value
        among all rows by (randomly) flipping True to False in other rows.
        """
        with torch.cuda.nvtx.range("cuda_tm/bool_align", color='yellow'):
            mask      = mask.to(torch.float32)
            true_cnts = mask.sum(dim=1)
            min_cnt   = int(true_cnts.amin().item())
            max_cnt   = int(true_cnts.amax().item())
            if min_cnt == max_cnt: return mask
            
            rand = torch.rand_like(mask, dtype=torch.float32) * mask
            topk = rand.topk(min_cnt, dim=1).indices

            new_mask = torch.zeros_like(mask, dtype=torch.bool)
            rows = torch.arange(mask.size(0), device=mask.device).unsqueeze(1).expand(-1, min_cnt)
            new_mask[rows, topk] = True
            
            return new_mask.to(torch.bool)

    @torch.no_grad()
    def precompute_contract_idx(self, aligned_mask: Jt.Bool[torch.Tensor, "B N//G"]):
        with torch.cuda.nvtx.range("cuda_tm/dstlidx", color='yellow'):
            dst_lidx = Extension.token_grpcontract_v2_scan(self.start_idx, self.grp_size, aligned_mask)
            return dst_lidx

    @torch.no_grad()
    def precompute_attn_mask(self, aligned_mask: Jt.Bool[torch.Tensor, "B N//G"], dst_lidx: torch.Tensor):
        with torch.cuda.nvtx.range("cuda_tm/attn_bias", color='yellow'):
            B, N = dst_lidx.size(0), int(dst_lidx[0, -1].item() + 1)
            dev  = dst_lidx.device
            
            dst_lidx_head = dst_lidx.reshape(B, -1, self.grp_size)[..., 0]
            start_idx     = dst_lidx_head[aligned_mask].reshape(B, -1)
            
            a_bias = torch.zeros((B, N), dtype=torch.float, device=dev)
            rows   = torch.arange(B, device=start_idx.device).unsqueeze(1).expand_as(start_idx)
            a_bias[rows, start_idx] = math.log(float(self.grp_size))

            return a_bias
