import torch
import cuda.tile as ct


@ct.kernel
def matrix_multiplication(
    a,
    b,
    output,
    TILE_M: ct.Constant[int],
    TILE_N: ct.Constant[int],
    TILE_K: ct.Constant[int],
):
    # Each CTA computes one TILE_M x TILE_N tile of the output matrix.
    tile_m = ct.bid(0)
    tile_n = ct.bid(1)
    accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)

    # Walk along the shared K dimension and accumulate the tile products.
    num_k_tiles = ct.num_tiles(a, axis=1, shape=(TILE_M, TILE_K))
    for tile_k in range(num_k_tiles):
        # Zero padding prevents boundary values from affecting the reduction.
        a_tile = ct.load(
            a,
            index=(tile_m, tile_k),
            shape=(TILE_M, TILE_K),
            padding_mode=ct.PaddingMode.ZERO,
        )
        b_tile = ct.load(
            b,
            index=(tile_k, tile_n),
            shape=(TILE_K, TILE_N),
            padding_mode=ct.PaddingMode.ZERO,
        )
        accumulator = ct.mma(a_tile, b_tile, accumulator)

    # Stores outside the output matrix are ignored for partial boundary tiles.
    ct.store(output, index=(tile_m, tile_n), tile=accumulator.astype(output.dtype))


def main():
    tile_m = 64
    tile_n = 64
    tile_k = 32

    # A is M x K, B is K x N; all dimensions exercise partial tiles.
    m = 4096 + 1
    n = 4096 + 2
    k = 4096 + 3

    # Prepare the inputs, expected result, and output with PyTorch.
    a = torch.rand(m, k, device="cuda", dtype=torch.float16)
    b = torch.rand(k, n, device="cuda", dtype=torch.float16)
    expect = a @ b
    output = torch.empty_like(expect)

    # Each grid position corresponds to one output tile.
    grid = (
        ct.cdiv(m, tile_m),
        ct.cdiv(n, tile_n),
        1,
    )
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid,
        matrix_multiplication,
        (a, b, output, tile_m, tile_n, tile_k),
    )
    torch.cuda.synchronize()

    # Floating-point accumulation order may differ from PyTorch.
    torch.testing.assert_close(output, expect, rtol=1e-3, atol=1e-2)
    print("matrix_multiplication passed")


if __name__ == "__main__":
    main()
