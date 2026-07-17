from itertools import product

import torch
import cuda.tile as ct


ENABLE_AUTOTUNE = False
FIXED_CONFIG = {
    "tile_m": 64,
    "tile_n": 128,
    "tile_k": 32,
    "occupancy": 4,
}


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


def matrix_multiplication_autotune(a, b, output):
    keys = ("tile_m", "tile_n", "tile_k", "occupancy")
    search_space = [
        dict(zip(keys, values))
        for values in product(
            (32, 64, 128),
            (32, 64, 128),
            (32, 64, 128),
            (1, 2, 4, 8, 16),
        )
    ]

    result = ct.tune.exhaustive_search(
        search_space=search_space,
        stream=torch.cuda.current_stream(),
        grid_fn=lambda cfg: (
            ct.cdiv(a.shape[0], cfg["tile_m"]),
            ct.cdiv(b.shape[1], cfg["tile_n"]),
            1,
        ),
        kernel=matrix_multiplication,
        # Each measurement gets a fresh output to prevent candidates from sharing state.
        args_fn=lambda cfg: (
            a,
            b,
            torch.empty_like(output),
            cfg["tile_m"],
            cfg["tile_n"],
            cfg["tile_k"],
        ),
        hints_fn=lambda cfg: {"occupancy": cfg["occupancy"]},
        quiet=True,
    )

    print(result)
    return result.best.config


def main():
    m = 4096
    n = 4096
    k = 4096

    # Prepare the inputs, expected result, and output with PyTorch.
    a = torch.rand(m, k, device="cuda", dtype=torch.float16)
    b = torch.rand(k, n, device="cuda", dtype=torch.float16)
    expect = a @ b

    output = torch.empty_like(expect)
    config = (
        matrix_multiplication_autotune(a, b, output)
        if ENABLE_AUTOTUNE
        else FIXED_CONFIG
    )
    
    # launch the kernel with the selected configuration.
    kernel = matrix_multiplication.replace_hints(
        occupancy=config["occupancy"]
    )
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        (
            ct.cdiv(m, config["tile_m"]),
            ct.cdiv(n, config["tile_n"]),
            1,
        ),
        kernel,
        (
            a,
            b,
            output,
            config["tile_m"],
            config["tile_n"],
            config["tile_k"],
        ),
    )
    torch.cuda.synchronize()

    # Floating-point accumulation order may differ from PyTorch.
    torch.testing.assert_close(output, expect, rtol=1e-3, atol=1e-2)
    mode = "autotuned" if ENABLE_AUTOTUNE else "fixed"
    print(f"GEMM with {mode} config passed: {config}")


if __name__ == "__main__":
    main()
