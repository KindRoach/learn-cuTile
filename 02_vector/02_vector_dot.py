import torch
import cuda.tile as ct


@ct.kernel
def vector_dot(a, b, output, TILE_SIZE: ct.Constant[int]):
    # Each CTA processes one tile of the vectors.
    tile_index = (ct.bid(0),)

    # Zero padding is required because padded values participate in the reduction.
    a_tile = ct.load(
        a,
        index=tile_index,
        shape=(TILE_SIZE,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    b_tile = ct.load(
        b,
        index=tile_index,
        shape=(TILE_SIZE,),
        padding_mode=ct.PaddingMode.ZERO,
    )

    # Reduce this tile, then safely accumulate partial results from all CTAs.
    partial_dot = ct.sum(a_tile * b_tile, axis=0)
    ct.atomic_add(output, 0, partial_dot)


def main():
    tile_size = 1024

    # Use a size that is not divisible by tile_size to exercise boundary handling.
    vector_size = 1024 * 1024 * 1024 + 123

    # Prepare the inputs and output on the GPU with PyTorch.
    a = torch.rand(vector_size, device="cuda", dtype=torch.float32)
    b = torch.rand(vector_size, device="cuda", dtype=torch.float32)
    expect = torch.dot(a, b)

    # Atomic accumulation requires the output to start at zero.
    output = torch.zeros(1, device="cuda", dtype=torch.float32)

    # Round up so the last CTA processes the remaining elements.
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid,
        vector_dot,
        (a, b, output, tile_size),
    )
    torch.cuda.synchronize()

    # Reduction order may differ, so compare floating-point results approximately.
    torch.testing.assert_close(
        output.squeeze(),
        expect,
        rtol=1e5,
        atol=1e-4,
    )
    print("vector_dot passed")


if __name__ == "__main__":
    main()
