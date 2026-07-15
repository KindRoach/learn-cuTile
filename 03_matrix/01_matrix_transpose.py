import torch
import cuda.tile as ct


@ct.kernel
def matrix_transpose(
    x,
    output,
    TILE_M: ct.Constant[int],
    TILE_N: ct.Constant[int],
):
    # Each CTA loads one tile from the input matrix.
    tile_m = ct.bid(0)
    tile_n = ct.bid(1)
    x_tile = ct.load(x, index=(tile_m, tile_n), shape=(TILE_M, TILE_N))

    # Transpose the tile locally and swap its tile indices in the output matrix.
    output_tile = ct.transpose(x_tile)
    ct.store(output, index=(tile_n, tile_m), tile=output_tile)


def main():
    tile_m = 128
    tile_n = 64

    # Use dimensions that are not divisible by the tile shape.
    m = 4096 + 1
    n = 4096 + 2

    # Prepare the input, expected result, and output with PyTorch.
    x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    expect = x.T.contiguous()
    output = torch.empty_like(expect)

    # Round up both grid dimensions to cover the partial boundary tiles.
    grid = (
        ct.cdiv(m, tile_m),
        ct.cdiv(n, tile_n),
        1,
    )
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid,
        matrix_transpose,
        (x, output, tile_m, tile_n),
    )
    torch.cuda.synchronize()

    # Verify the result against PyTorch.
    torch.testing.assert_close(output, expect)
    print("matrix_transpose passed")


if __name__ == "__main__":
    main()
