import torch
import numpy as np
import cuda.tile as ct


@ct.kernel
def vector_add(a, b, c, tile_size: ct.Constant[int]):
    pid = ct.bid(0)

    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))

    result = a_tile + b_tile

    ct.store(c, index=(pid,), tile=result)


def test():
    vector_size = 2**12
    tile_size = 2**4
    grid = (ct.cdiv(vector_size, tile_size), 1, 1)

    device = "cuda"

    a = torch.rand(vector_size, device=device)
    b = torch.rand(vector_size, device=device)
    c = torch.zeros_like(a)

    # Launch kernel
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        grid,
        vector_add,
        (a, b, c, tile_size),
    )

    # Synchronize before checking
    torch.cuda.synchronize()

    a_np = a.cpu().numpy()
    b_np = b.cpu().numpy()
    c_np = c.cpu().numpy()

    expected = a_np + b_np
    np.testing.assert_allclose(c_np, expected)

    print("✓ vector_add_example passed!")


if __name__ == "__main__":
    test()