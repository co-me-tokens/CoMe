import pytest
import torch

import jaxtyping as jt
from dataclasses import dataclass

from src.accelerate.token_merger.co_me import func_token_merge, func_token_split
from src.utility.feature import CUDAExtension


@dataclass(kw_only=True)
class Datapoint:
    input_tokens: jt.Float[torch.Tensor, "B N C"]
    input_merge_mask  : jt.Bool[torch.Tensor, "B M"]
    input_start_index: int
    input_group_size : int
    
    output_tokens: jt.Float[torch.Tensor, "1 P C"]
    output_offset: jt.Float[torch.Tensor, "B+1"]
    output_revidx: jt.Float[torch.Tensor, "B*N"]
    output_weight: jt.Float[torch.Tensor, "P"]
    
    restore_tokens: jt.Float[torch.Tensor, "B N C"]


manual_designed_cases: list[Datapoint] = [
    Datapoint(
        input_tokens=torch.tensor([
            [0., 1., 2., 3. , 4. , 5. , 6. ], 
            [7., 8., 9., 10., 11., 12., 13.],
        ]).unsqueeze(-1),
        input_merge_mask=torch.tensor([
            [True , False, True], [False, False, True],
        ], dtype=torch.bool),
        input_start_index=1,
        input_group_size=2,
        output_tokens=torch.tensor([
            [0., 1.5, 3., 4., 5.5, 7., 8., 9., 10., 11., 12.5]
        ]).unsqueeze(-1),
        output_offset=torch.tensor([0, 5, 11], dtype=torch.long),
        output_revidx=torch.tensor([
            [0, 1, 1, 2, 3, 4 , 4 ],
            [5, 6, 7, 8, 9, 10, 10]
        ], dtype=torch.long).flatten(),
        output_weight=torch.tensor(
            [1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2],
            dtype=torch.float
        ),
        restore_tokens=torch.tensor([
            [0, 1.5, 1.5, 3. , 4. , 5.5 , 5.5 ],
            [7., 8., 9. , 10., 11., 12.5, 12.5]
        ]).unsqueeze(-1)
    ),
    Datapoint(
        input_tokens=torch.tensor([
            [1., 2., 3. , 4. , 5. , 6. ], 
            [8., 9., 10., 11., 12., 13.],
        ]).unsqueeze(-1),
        input_merge_mask=torch.tensor([
            [True , False, True], [False, False, True],
        ], dtype=torch.bool),
        input_start_index=0,
        input_group_size=2,
        output_tokens=torch.tensor([
            [1.5, 3., 4., 5.5, 8., 9., 10., 11., 12.5]
        ]).unsqueeze(-1),
        output_offset=torch.tensor([0, 4, 9], dtype=torch.long),
        output_revidx=torch.tensor([
            [0, 0, 1, 2, 3, 3],
            [4, 5, 6, 7, 8, 8]
        ], dtype=torch.long).flatten(),
        output_weight=torch.tensor(
            [2, 1, 1, 2, 1, 1, 1, 1, 2], dtype=torch.float
        ),
        restore_tokens=torch.tensor([
            [1.5, 1.5, 3. , 4. , 5.5 , 5.5 ],
            [8. , 9. , 10., 11., 12.5, 12.5]
        ]).unsqueeze(-1)
    )
]

ai_generated_test_cases: list[Datapoint] = [
    # Case 1: All groups merged
    Datapoint(
        input_tokens=torch.tensor([[1., 2., 3., 4., 5., 6.]]).unsqueeze(-1),
        input_merge_mask=torch.tensor([[True, True, True]]),
        input_start_index=0,
        input_group_size=2,
        output_tokens=torch.tensor([[1.5, 3.5, 5.5]]).unsqueeze(-1),
        output_offset=torch.tensor([0, 3], dtype=torch.long),
        output_revidx=torch.tensor([[0, 0, 1, 1, 2, 2]], dtype=torch.long).flatten(),
        output_weight=torch.tensor([2, 2, 2], dtype=torch.float),
        restore_tokens=torch.tensor([[1.5, 1.5, 3.5, 3.5, 5.5, 5.5]]).unsqueeze(-1),
    ),
    # Case 2: No groups merged
    Datapoint(
        input_tokens=torch.tensor([[1., 2., 3., 4., 5., 6.]]).unsqueeze(-1),
        input_merge_mask=torch.tensor([[False, False, False]]),
        input_start_index=0,
        input_group_size=2,
        output_tokens=torch.tensor([[1., 2., 3., 4., 5., 6.]]).unsqueeze(-1),
        output_offset=torch.tensor([0, 6], dtype=torch.long),
        output_revidx=torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long).flatten(),
        output_weight=torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float),
        restore_tokens=torch.tensor([[1., 2., 3., 4., 5., 6.]]).unsqueeze(-1),
    ),
    # Case 3: group_size=3
    Datapoint(
        input_tokens=torch.tensor([[1., 2., 3., 4., 5., 6.]]).unsqueeze(-1),
        input_merge_mask=torch.tensor([[True, False]]),
        input_start_index=0,
        input_group_size=3,
        output_tokens=torch.tensor([[2., 4., 5., 6.]]).unsqueeze(-1),
        output_offset=torch.tensor([0, 4], dtype=torch.long),
        output_revidx=torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.long).flatten(),
        output_weight=torch.tensor([3, 1, 1, 1], dtype=torch.float),
        restore_tokens=torch.tensor([[2., 2., 2., 4., 5., 6.]]).unsqueeze(-1),
    ),
    # Case 4: Multi-channel (C=2)
    Datapoint(
        input_tokens=torch.tensor([[[1., 10.], [2., 20.], [3., 30.], [4., 40.]]]),
        input_merge_mask=torch.tensor([[True, False]]),
        input_start_index=0,
        input_group_size=2,
        output_tokens=torch.tensor([[[1.5, 15.], [3., 30.], [4., 40.]]]),
        output_offset=torch.tensor([0, 3], dtype=torch.long),
        output_revidx=torch.tensor([[0, 0, 1, 2]], dtype=torch.long).flatten(),
        output_weight=torch.tensor([2, 1, 1], dtype=torch.float),
        restore_tokens=torch.tensor([[[1.5, 15.], [1.5, 15.], [3., 30.], [4., 40.]]]),
    ),
    # Case 5: Three batches with varied merge patterns
    Datapoint(
        input_tokens=torch.tensor([
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
        ]).unsqueeze(-1),
        input_merge_mask=torch.tensor([
            [True, True], [False, False], [True, False],
        ]),
        input_start_index=0,
        input_group_size=2,
        output_tokens=torch.tensor([
            [1.5, 3.5, 5., 6., 7., 8., 9.5, 11., 12.]
        ]).unsqueeze(-1),
        output_offset=torch.tensor([0, 2, 6, 9], dtype=torch.long),
        output_revidx=torch.tensor([
            [0, 0, 1, 1],
            [2, 3, 4, 5],
            [6, 6, 7, 8],
        ], dtype=torch.long).flatten(),
        output_weight=torch.tensor([2, 2, 1, 1, 1, 1, 2, 1, 1], dtype=torch.float),
        restore_tokens=torch.tensor([
            [1.5, 1.5, 3.5, 3.5],
            [5.0, 6.0, 7.0, 8.0],
            [9.5, 9.5, 11.0, 12.0],
        ]).unsqueeze(-1),
    ),
    # Case 6: Large group_size=4 with special tokens
    Datapoint(
        input_tokens=torch.tensor([
            [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.],
            [10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        ]).unsqueeze(-1),
        input_merge_mask=torch.tensor([
            [True, False], [False, True],
        ]),
        input_start_index=2,
        input_group_size=4,
        output_tokens=torch.tensor([
            [0., 1., 3.5, 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 17.5]
        ]).unsqueeze(-1),
        output_offset=torch.tensor([0, 7, 14], dtype=torch.long),
        output_revidx=torch.tensor([
            [0, 1, 2, 2, 2, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12, 13, 13, 13, 13],
        ], dtype=torch.long).flatten(),
        output_weight=torch.tensor(
            [1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4], dtype=torch.float
        ),
        restore_tokens=torch.tensor([
            [0.0, 1.0, 3.5, 3.5, 3.5, 3.5, 6.0, 7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 17.5, 17.5, 17.5, 17.5],
        ]).unsqueeze(-1),
    ),
    # Case 7: group_size=1 (merging a single token is identity)
    Datapoint(
        input_tokens=torch.tensor([[10., 20., 30., 40.]]).unsqueeze(-1),
        input_merge_mask=torch.tensor([[True, False, True]]),
        input_start_index=1,
        input_group_size=1,
        output_tokens=torch.tensor([[10., 20., 30., 40.]]).unsqueeze(-1),
        output_offset=torch.tensor([0, 4], dtype=torch.long),
        output_revidx=torch.tensor([[0, 1, 2, 3]], dtype=torch.long).flatten(),
        output_weight=torch.tensor([1, 1, 1, 1], dtype=torch.float),
        restore_tokens=torch.tensor([[10., 20., 30., 40.]]).unsqueeze(-1),
    )
]


def _to_cuda(case: Datapoint) -> Datapoint:
    return Datapoint(
        input_tokens=case.input_tokens.cuda(),
        input_merge_mask=case.input_merge_mask.cuda(),
        input_start_index=case.input_start_index,
        input_group_size=case.input_group_size,
        output_tokens=case.output_tokens.cuda(),
        output_offset=case.output_offset.cuda(),
        output_revidx=case.output_revidx.cuda(),
        output_weight=case.output_weight.cuda(),
        restore_tokens=case.restore_tokens.cuda(),
    )


CUDA_TEST_CASES = [_to_cuda(c) for c in manual_designed_cases + ai_generated_test_cases]


@pytest.mark.parametrize(argnames=["case"], argvalues=[
    (case,) for case in CUDA_TEST_CASES
])
def test_token_merge_func(case: Datapoint):
    merged_tokens, rev_index = func_token_merge(
        tokens     =case.input_tokens,
        merge_mask =case.input_merge_mask,
        start_index=case.input_start_index, group_size=case.input_group_size, 
    )
    
    torch.testing.assert_close(merged_tokens.tokens, case.output_tokens)
    torch.testing.assert_close(merged_tokens.offset, case.output_offset)
    torch.testing.assert_close(merged_tokens.weight, case.output_weight)
    torch.testing.assert_close(rev_index           , case.output_revidx)
    

@pytest.mark.parametrize(argnames=["case"], argvalues=[
    (case,) for case in CUDA_TEST_CASES
])
def test_token_split_func(case: Datapoint):
    merged_tokens, rev_index = func_token_merge(
        tokens     =case.input_tokens,
        merge_mask =case.input_merge_mask,
        start_index=case.input_start_index, group_size=case.input_group_size, 
    )
    torch.testing.assert_close(merged_tokens.tokens, case.output_tokens)
    torch.testing.assert_close(merged_tokens.offset, case.output_offset)
    torch.testing.assert_close(merged_tokens.weight, case.output_weight)
    torch.testing.assert_close(rev_index           , case.output_revidx)
    
    s, n, c = case.input_tokens.shape
    restored_tokens = func_token_split(merged_tokens, rev_index, (1, s, n, c))
    torch.testing.assert_close(restored_tokens, case.restore_tokens)


@pytest.mark.parametrize(argnames=["case"], argvalues=[
    (case,) for case in CUDA_TEST_CASES
])
def test_token_merge_cuda_pytorch_parity(case: Datapoint):
    merged_tokens_cuda, rev_index_cuda = func_token_merge(
        tokens=case.input_tokens,
        merge_mask=case.input_merge_mask,
        start_index=case.input_start_index, group_size=case.input_group_size,
    )
    with CUDAExtension.disable():
        merged_tokens_pytorch, rev_index_pytorch = func_token_merge(
            tokens=case.input_tokens,
            merge_mask=case.input_merge_mask,
            start_index=case.input_start_index, group_size=case.input_group_size,
        )

    torch.testing.assert_close(merged_tokens_cuda.tokens, merged_tokens_pytorch.tokens)
    torch.testing.assert_close(merged_tokens_cuda.offset, merged_tokens_pytorch.offset)
    torch.testing.assert_close(merged_tokens_cuda.weight, merged_tokens_pytorch.weight)
    torch.testing.assert_close(rev_index_cuda, rev_index_pytorch)


@pytest.mark.parametrize(argnames=["case"], argvalues=[
    (case,) for case in CUDA_TEST_CASES
])
def test_token_split_cuda_pytorch_parity(case: Datapoint):
    merged_tokens_cuda, rev_index_cuda = func_token_merge(
        tokens=case.input_tokens,
        merge_mask=case.input_merge_mask,
        start_index=case.input_start_index, group_size=case.input_group_size,
    )
    s, n, c = case.input_tokens.shape
    restored_tokens_cuda = func_token_split(merged_tokens_cuda, rev_index_cuda, (1, s, n, c))

    with CUDAExtension.disable():
        merged_tokens_pytorch, rev_index_pytorch = func_token_merge(
            tokens=case.input_tokens,
            merge_mask=case.input_merge_mask,
            start_index=case.input_start_index, group_size=case.input_group_size,
        )
        restored_tokens_pytorch = func_token_split(merged_tokens_pytorch, rev_index_pytorch, (1, s, n, c))

    torch.testing.assert_close(restored_tokens_cuda, restored_tokens_pytorch)


@pytest.mark.parametrize(argnames=["tokens", "merge_mask", "start_index", "group_size", "expected_msg"], argvalues=[
    # N != start_index + M * group_size
    (torch.randn(2, 7, 4), torch.ones(2, 3, dtype=torch.bool), 0, 2, "Expected N = start_index"),
    # merge_mask batch != tokens batch
    (torch.randn(2, 6, 4), torch.ones(3, 3, dtype=torch.bool), 0, 2, "merge_mask batch"),
    # start_index < 0
    (torch.randn(1, 6, 4), torch.ones(1, 3, dtype=torch.bool), -1, 2, "start_index"),
    # group_size <= 0
    (torch.randn(1, 6, 4), torch.ones(1, 3, dtype=torch.bool), 0, 0, "group_size"),
    # merge_mask not bool
    (torch.randn(1, 6, 4), torch.ones(1, 3, dtype=torch.float32), 0, 2, "torch.bool"),
])
def test_token_merge_validation(
    tokens: torch.Tensor,
    merge_mask: torch.Tensor,
    start_index: int,
    group_size: int,
    expected_msg: str,
):
    with pytest.raises(ValueError, match=expected_msg):
        func_token_merge(tokens=tokens, merge_mask=merge_mask, start_index=start_index, group_size=group_size)
