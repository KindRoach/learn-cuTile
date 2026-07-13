"""Infer an AOT signature from PyTorch tensors and export a cuTile kernel.

Run:
    python 01_basic/02_export_kernel.py

Outputs:
    01_basic/export_kernel_output/add_one.cubin
    01_basic/export_kernel_output/add_one.tileir

This basic example infers constraints from example tensors for convenience. Production
code should use explicit constraints to avoid capturing accidental alignment assumptions.
"""

from pathlib import Path

import torch
import cuda.tile as ct
from cuda.tile import compilation as cc


TILE_SIZE = 128
OUTPUT_DIR = Path(__file__).with_name("export_kernel_output")


@ct.kernel
def add_one(x, output):
    """Process one tile of 128 float32 elements per CTA."""
    tile_index = (ct.bid(0),)
    values = ct.load(x, tile_index, (TILE_SIZE,))
    ct.store(output, tile_index, values + 1.0)


def current_gpu_code() -> str:
    """Convert the compute capability reported by PyTorch to sm_XX."""
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("AOT export requires an available CUDA GPU")

    # These tensors are inspected only to infer the AOT signature. Their data is not
    # embedded in the exported files. Use separate tensors because x and output must
    # not alias each other.
    x = torch.empty(2 * TILE_SIZE, device="cuda", dtype=torch.float32)
    output = torch.empty_like(x)
    signature = cc.KernelSignature.from_kernel_args(
        add_one,
        kernel_args=(x, output),
        calling_convention=cc.CallingConvention.cutile_python_v1(),
        symbol="add_one_f32",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gpu_code = current_gpu_code()

    outputs = (
        (OUTPUT_DIR / "add_one.cubin", "cubin"),
        (OUTPUT_DIR / "add_one.tileir", "tileir_bytecode"),
    )
    for output_file, output_format in outputs:
        cc.export_kernel(
            add_one,
            [signature],
            output_file,
            gpu_code=gpu_code,
            output_format=output_format,
        )
        print(f"Exported {output_format}: {output_file}")

    print(f"Target GPU: {gpu_code}")
    print("Exported symbol: add_one_f32")
    print("ABI: x_ptr, x_shape, x_stride, output_ptr, output_shape, output_stride")


if __name__ == "__main__":
    main()
