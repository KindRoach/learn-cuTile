import torch
import cuda.tile as ct


@ct.kernel
def hello_world():
    # print once for each CTA.
    ct.print(f"Hello, world from CTA {ct.bid(0)}")


def main():
    # Launch four CTAs: threads per CTA decieded by cuTile compliler
    ct.launch(
        torch.cuda.current_stream().cuda_stream,
        (4, 1, 1),
        hello_world,
        (),
    )
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
