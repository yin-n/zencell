import argparse
import os

import models_vit
import numpy as np
import torch
import zarr
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=False)
parser.add_argument("--ckpt", type=str, required=False)
parser.add_argument("--output_dir", type=str, required=False)
parser.add_argument(
    "--chns",
    type=lambda x: tuple(int(y) for y in x.split(",")),
    nargs="+",
    required=False,
)
parser.add_argument("--zarr_path", type=str, required=False)
args = parser.parse_args()

## add parameters for debugging
args.model = (
    "vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale"
)
args.ckpt = "/mnt/aperto/yin/napari_cellseg/napari_cellseg3d/code_models/models/zencell/checkpoint-4999.pth"
args.output_dir = "/mnt/aperto/yin/napari_cellseg/napari_cellseg3d/code_models/models/zencell/output"
args.chns = [(3, 1)]
args.zarr_path = "/mnt/aperto/tatz_brain_data/240620_01_MX007-1/fused.zarr"


rank = int(os.environ["SLURM_PROCID"])
world_size = int(os.environ["SLURM_NPROCS"])
torch.cuda.set_device(rank % torch.cuda.device_count())


model = models_vit.__dict__[args.model](out_chans=4)
model.half().eval().cuda()
model.load_state_dict(
    torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
)

# TODO: change z size due to the memory limitation
# z, y, x = 40, 1024, 1024


# z, y, x = 1000, 1024, 1024

z, y, x = 40, 1024, 1024


zarr_fp = zarr.open(args.zarr_path, mode="r")
zmax, ymax, xmax = zarr_fp["0"].shape[1:]
zmax, ymax, xmax = 1240, 5024, 4024
print(f"zmax: {zmax}, ymax: {ymax}, xmax: {xmax}")
# zmax, ymax, xmax = zarr_fp["setup3"]["timepoint0"]["s0"].shape
task_list = []
# for ref_chn, sig_chn in [(3, 1), (8, 7), (11, 9), (11, 10)]:
# for ref_chn, sig_chn in args.chns:
#     for z0 in range(0, zmax, z):
#         for y0 in range(0, ymax, y):
#             for x0 in range(0, xmax, x):
#                 task_list.append((ref_chn, sig_chn, z0, y0, x0))

for ref_chn, sig_chn in args.chns:
    for z0 in range(1200, zmax, z):
        for y0 in range(4000, ymax, y):
            for x0 in range(3000, xmax, x):
                task_list.append((ref_chn, sig_chn, z0, y0, x0))
if rank == 0:
    print(f"Total task: {len(task_list)}")

rz, ry, rx = model.patch_embed_roi.img_size
cz, cy, cx = model.patch_embed_ctx.img_size
az, ay, ax = model.patch_embed_ctx.patch_size

print(f"rz: {rz}, ry: {ry}, rx: {rx}")
print(f"z: {z}, y: {y}, x: {x}")
assert z % rz == 0 and y % ry == 0 and x % rx == 0
zpad_head, ypad_head, xpad_head = (
    (cz - rz) // 2 // az * az,
    (cy - ry) // 2 // ay * ay,
    (cx - rx) // 2 // ax * ax,
)
zpad_tail, ypad_tail, xpad_tail = (
    cz - rz - zpad_head,
    cy - ry - ypad_head,
    cx - rx - xpad_head,
)


def crop_with_pad(chn, z0, y0, x0):
    z_st, z_ed = z0 - zpad_head - rz // 2, z0 + z + zpad_tail + rz // 2
    y_st, y_ed = y0 - ypad_head - ry // 2, y0 + y + ypad_tail + ry // 2
    x_st, x_ed = x0 - xpad_head - rx // 2, x0 + x + xpad_tail + rx // 2
    z_slice = slice(max(z_st, 0), min(z_ed, zmax))
    y_slice = slice(max(y_st, 0), min(y_ed, ymax))
    x_slice = slice(max(x_st, 0), min(x_ed, xmax))
    z_pad = (max(-z_st, 0), max(z_ed - zmax, 0))
    y_pad = (max(-y_st, 0), max(y_ed - ymax, 0))
    x_pad = (max(-x_st, 0), max(x_ed - xmax, 0))
    arr = zarr_fp["0"][chn, z_slice, y_slice, x_slice]
    # arr = zarr_fp[f"setup{chn}"]["timepoint0"]["s0"][z_slice, y_slice, x_slice]
    if any(a > 0 or b > 0 for a, b in (z_pad, y_pad, x_pad)):
        arr = np.pad(arr, (z_pad, y_pad, x_pad))
    return arr


for ref_chn, sig_chn, z0, y0, x0 in (tqdm if rank == 0 else lambda x: x)(
    task_list[rank::world_size]
):
    ref_arr = crop_with_pad(ref_chn, z0, y0, x0)
    sig_arr = crop_with_pad(sig_chn, z0, y0, x0)

    z_offs_list, y_offs_list, x_offs_list = np.meshgrid(
        np.arange(0, z + 1, rz // 2),
        np.arange(0, y + 1, ry // 2),
        np.arange(0, x + 1, rx // 2),
    )
    z_offs_list, y_offs_list, x_offs_list = (
        z_offs_list.reshape(-1).tolist(),
        y_offs_list.reshape(-1).tolist(),
        x_offs_list.reshape(-1).tolist(),
    )

    cell_prob = torch.zeros([z + rz, y + ry, x + rx], device="cuda")

    print("cellpob size is: ", cell_prob.size())
    cell_flow = torch.zeros([3, z + rz, y + ry, x + rx], device="cuda")
    weight = torch.zeros([z + rz, y + ry, x + rx], device="cuda")
    z_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=rz) - 1.0) ** 2
    y_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=ry) - 1.0) ** 2
    x_dist_sq = (torch.linspace(start=0.0, end=2.0, steps=rx) - 1.0) ** 2
    weight_template = (
        3.0
        - z_dist_sq.view(-1, 1, 1)
        - y_dist_sq.view(1, -1, 1)
        - x_dist_sq.view(1, 1, -1)
    ) / 3.0
    weight_template = weight_template.cuda()

    with torch.no_grad():
        ref_arr = torch.from_numpy(ref_arr).cuda()
        sig_arr = torch.from_numpy(sig_arr).cuda()
        for z_offs, y_offs, x_offs in zip(
            z_offs_list, y_offs_list, x_offs_list, strict=False
        ):
            ref_slice = ref_arr[
                z_offs : z_offs + cz,
                y_offs : y_offs + cy,
                x_offs : x_offs + cx,
            ]
            sig_slice = sig_arr[
                z_offs : z_offs + cz,
                y_offs : y_offs + cy,
                x_offs : x_offs + cx,
            ]
            # ref_slice = torch.from_numpy(ref_slice).cuda()
            # sig_slice = torch.from_numpy(sig_slice).cuda()
            input_vol = torch.stack([ref_slice, sig_slice], dim=0).float()
            assert input_vol.size() == (2, cz, cy, cx)
            input_vol = input_vol - input_vol.mean(dim=(1, 2, 3), keepdim=True)
            input_vol = input_vol / (
                input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6
            )
            input_vol = input_vol.half()
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
            ):
                slice_pred = model(input_vol[None])[0]
            assert slice_pred.size()[1:] == (rz, ry, rx)
            cell_prob[
                z_offs : z_offs + rz,
                y_offs : y_offs + ry,
                x_offs : x_offs + rx,
            ] += (
                slice_pred[0].sigmoid() * weight_template
            )
            cell_flow[
                :,
                z_offs : z_offs + rz,
                y_offs : y_offs + ry,
                x_offs : x_offs + rx,
            ] += (
                slice_pred[1:] * weight_template
            )
            weight[
                z_offs : z_offs + rz,
                y_offs : y_offs + ry,
                x_offs : x_offs + rx,
            ] += weight_template
    weight += 1e-12
    cell_prob /= weight
    cell_flow /= weight
    cell_prob = (
        cell_prob.cpu()[
            rz // 2 : -rz // 2, ry // 2 : -ry // 2, rx // 2 : -rx // 2
        ]
        .contiguous()
        .numpy()
    )
    cell_flow = (
        cell_flow.cpu()[
            :, rz // 2 : -rz // 2, ry // 2 : -ry // 2, rx // 2 : -rx // 2
        ]
        .contiguous()
        .numpy()
    )

    np.save(
        os.path.join(
            args.output_dir,
            f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_prob.npy",
        ),
        cell_prob,
    )
    np.save(
        os.path.join(
            args.output_dir,
            f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_flow.npy",
        ),
        cell_flow,
    )
