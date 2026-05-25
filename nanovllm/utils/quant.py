import torch


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def dequant_fp8_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    row_start: int = 0,
    col_start: int = 0,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    block_n, block_k = block_size
    rows, cols = weight.shape
    scale_row_start = row_start // block_n
    scale_col_start = col_start // block_k
    scale_row_end = _ceil_div(row_start + rows, block_n)
    scale_col_end = _ceil_div(col_start + cols, block_k)
    scale = scale_inv[scale_row_start:scale_row_end, scale_col_start:scale_col_end]
    scale = scale.repeat_interleave(block_n, 0).repeat_interleave(block_k, 1)
    row_offset = row_start % block_n
    col_offset = col_start % block_k
    scale = scale[row_offset:row_offset + rows, col_offset:col_offset + cols]
    return (weight.float() * scale.float()).to(torch.bfloat16)


def maybe_dequant_fp8_weight(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None,
    row_start: int = 0,
    col_start: int = 0,
) -> torch.Tensor:
    if scale_inv is None:
        return weight
    return dequant_fp8_weight(weight, scale_inv, row_start, col_start)
