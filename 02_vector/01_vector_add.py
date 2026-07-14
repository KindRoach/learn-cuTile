import torch
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, output, TILE_SIZE: ct.Constant[int]):
    # Each CTA processes one tile of the vectors.
    tile_index = (ct.bid(0),)

    # The default padding mode is UNDETERMINED. This is safe here because
    # out-of-bounds values never affect in-bounds elements, and store ignores them.
    a_tile = ct.load(a, index=tile_index, shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=tile_index, shape=(TILE_SIZE,))

    # Addition is applied elementwise across the tile.
    sum = a_tile + b_tile
    ct.store(output, index=tile_index, tile=sum)


def main():
    tile_size = 1024

    # Use a size that is not divisible by tile_size to exercise boundary handling.
    vector_size = 1024 * 1024 * 1024 + 123

    # Prepare the inputs and output on the GPU with PyTorch.
    a = torch.rand(vector_size, device="cuda", dtype=torch.float32)
    b = torch.rand(vector_size, device="cuda", dtype=torch.float32)
    expect = a + b
    output = torch.empty_like(expect)

    # Round up so the last CTA processes the remaining elements.
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid,
        vector_add,
        (a, b, output, tile_size),
    )
    torch.cuda.synchronize()

    # Verify the result against PyTorch.
    torch.testing.assert_close(output, expect)
    print("vector_add passed")


if __name__ == "__main__":
    main()
