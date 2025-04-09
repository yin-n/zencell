import dask.array as da
import numpy as np
import torch
import zarr

from zencell.zencell_model.cellpose.dynamics import compute_masks


def remap(org_mask, stitch_mask):
    cm1 = torch.tensor(stitch_mask, device="cuda")
    cm2 = torch.tensor(org_mask, device="cuda")
    cm12 = ((cm1.long() << 32) + cm2.long()).to(torch.int64)
    keys, values = torch.unique(cm12.long(), return_counts=True)
    area_dict = {
        k: v for k, v in zip(keys.tolist(), values.tolist(), strict=False)
    }
    iou_dict = {k: {} for k in np.unique(org_mask)[1:]}

    for k in area_dict:
        id2 = k & 0xFFFFFFFF  ### original
        id1 = k >> 32  ### stitch
        if id1 != 0 and id2 != 0:
            iou_dict[id2][id1] = area_dict[k] / (
                area_dict[k]
                + area_dict.get(id2, 0)
                + area_dict.get(id1 << 32, 0)
            )

    ### record the stitch cells
    overlap_id = {k: [] for k in np.unique(stitch_mask)[1:]}
    id_map = np.zeros([org_mask.max() + 1], dtype=np.int64)

    for i in np.unique(org_mask)[1:]:
        id_map[i] = i
        iou_and_id1_list = [(v, id1) for id1, v in iou_dict[i].items()]
        if len(iou_and_id1_list) == 0:
            pass
        else:
            max_iou, id1 = max(iou_and_id1_list)
            overlap_id[id1].append(
                i
            )  # key is the stitch, value is the orginal mask

    for value in overlap_id.values():
        if len(value) > 1:
            #### need change the label
            relabel = np.min(value)
            for i in value:
                id_map[i] = relabel

    id_map = torch.tensor(id_map, device="cuda")
    remap_mask = id_map[cm2.long()].to(torch.int32).cpu().numpy()
    return remap_mask


def loadmask(zarrfile, mask, zst, zed, yst, yed, xst, xed):
    D, H, W = mask.shape
    for d in range(0, D, 500):
        for y in range(0, H, 512):
            for x in range(0, W, 512):
                zarrfile[
                    zst + d : min(zst + d + 500, zed),
                    yst + y : min(yst + y + 512, yed),
                    xst + x : min(xst + x + 512, xed),
                ] = mask[
                    d : min(d + 500, zed - zst),
                    y : min(y + 512, yed - yst),
                    x : min(x + 512, xed - xst),
                ]


if __name__ == "__main__":
    zarr_fp = zarr.open(
        "/home/share/zlin/tatz_brain_data/240620_01_MX007-1/fused.zarr",
        mode="r",
    )["0"]
    dask_arr = da.from_zarr(zarr_fp)
    sig = 1
    ref = 3
    D, H, W = dask_arr[sig].shape
    print(f"allocating cellmask Sig{sig}")
    ### create the whole brain mask as zarr
    zarr_file = zarr.group(
        store=zarr.N5Store(
            f"/data/share/mxia/Tatz/whole_brain_probability/whole_brain_250131/brain1/sig{sig}_ref{ref}/whole_brain_mask_fused.n5"
        )
    )
    shape = (D, H, W)
    chunks = (500, 512, 512)
    if "data" not in zarr_file:
        wholebrain = zarr_file.create_dataset(
            "data",
            shape=shape,
            chunks=chunks,
            dtype="int32",
            # compressor=zarr.Blosc(cname='zstd', clevel=3)
        )
    else:
        wholebrain = zarr_file["data"]

    #### stitch ####
    max_cell_id = 0
    print("Begin Stitching")

    z_prev_prob = np.empty([20, H, W], dtype=np.float32)
    z_prev_flow = np.empty([3, 20, H, W], dtype=np.float32)
    z_curr_prob = np.empty([20, H, W], dtype=np.float32)
    z_curr_flow = np.empty([3, 20, H, W], dtype=np.float32)
    z_next_prob = np.empty([20, H, W], dtype=np.float32)
    z_next_flow = np.empty([3, 20, H, W], dtype=np.float32)

    for z in range(0, D, 1000):
        ### consider gap alone z ###
        z_sz = min(1000, D - z)

        y_prev_prob = np.empty([z_sz, 20, W], dtype=np.float32)
        y_prev_flow = np.empty([3, z_sz, 20, W], dtype=np.float32)
        y_curr_prob = np.empty([z_sz, 20, W], dtype=np.float32)
        y_curr_flow = np.empty([3, z_sz, 20, W], dtype=np.float32)
        y_next_prob = np.empty([z_sz, 20, W], dtype=np.float32)
        y_next_flow = np.empty([3, z_sz, 20, W], dtype=np.float32)

        for y in range(0, H, 1024):
            ### consider gap alone y ###
            y_sz = min(1024, H - y)
            for x in range(0, W, 1024):
                print(f"Stitch z{z:04} y{y:04} x{x:04}", max_cell_id)
                ### consider gap alone x ###

                ### load mask, update mask based on max_cell_id and update the max_cell_id ###
                mask_path = f"/data/share/zlin/mae_cell_ctx/output_dir/blocks_250115_2_brain1/sig{sig}_ref{ref}_z{z:04}_y{y:04}_x{x:04}_cellmask.npy"
                mask = np.load(mask_path).astype(np.int32)
                current_max_cell_id = mask.max()
                mask[np.nonzero(mask)] += max_cell_id
                max_cell_id += current_max_cell_id
                print("Mask max", mask.max())
                ### put the mask into the zarr file ###
                x_sz = min(1024, W - x)

                ### load the mask into the zarr file ###
                loadmask(
                    wholebrain, mask, z, z + z_sz, y, y + y_sz, x, x + x_sz
                )

                ### load the prob and flow ###
                prob = np.load(
                    f"/data/share/zlin/mae_cell_ctx/output_dir/blocks_250115_2_brain1/sig{sig}_ref{ref}_z{z:04}_y{y:04}_x{x:04}_cell_prob.npy"
                )
                flow = np.load(
                    f"/data/share/zlin/mae_cell_ctx/output_dir/blocks_250115_2_brain1/sig{sig}_ref{ref}_z{z:04}_y{y:04}_x{x:04}_cell_flow.npy"
                )

                if x == 0:
                    x_prev_prob = prob[:z_sz, :y_sz, -20:]
                    x_prev_flow = flow[:, :z_sz, :y_sz, -20:]
                else:
                    x_curr_prob = prob[:z_sz, :y_sz, :20]
                    x_curr_flow = flow[:, :z_sz, :y_sz, :20]

                    stitch_prob = np.concatenate(
                        (x_prev_prob, x_curr_prob), axis=-1
                    )
                    if (stitch_prob < 0.5).all():
                        pass
                    else:
                        stitch_flow = np.concatenate(
                            (x_prev_flow, x_curr_flow), axis=-1
                        )
                        stitch_mask, _ = compute_masks(
                            stitch_flow,
                            stitch_prob,
                            min_size=0,
                            flow_threshold=None,
                            cellprob_threshold=0.5,
                            do_3D=True,
                        )
                        org_mask = wholebrain[
                            z : z + z_sz, y : y + y_sz, x - 20 : x + 20
                        ]
                        remap_mask = remap(org_mask, stitch_mask)
                        loadmask(
                            wholebrain,
                            remap_mask,
                            z,
                            z + z_sz,
                            y,
                            y + y_sz,
                            x - 20,
                            x + 20,
                        )

                    ### current prob and flow will become next prev ###
                    x_prev_prob = prob[:z_sz, :y_sz, -20:].copy()
                    x_prev_flow = flow[:, :z_sz, :y_sz, -20:].copy()

                if y == 0:
                    y_prev_prob[:, :, x : x + x_sz] = prob[:z_sz, -20:, :x_sz]
                    y_prev_flow[:, :, :, x : x + x_sz] = flow[
                        :, :z_sz, -20:, :x_sz
                    ]
                else:
                    y_curr_prob[:, :, x : x + x_sz] = prob[:z_sz, :20, :x_sz]
                    y_curr_flow[:, :, :, x : x + x_sz] = flow[
                        :, :z_sz, :20, :x_sz
                    ]
                    y_next_prob[:, :, x : x + x_sz] = prob[:z_sz, -20:, :x_sz]
                    y_next_flow[:, :, :, x : x + x_sz] = flow[
                        :, :z_sz, -20:, :x_sz
                    ]

                if z == 0:
                    z_prev_prob[:, y : y + y_sz, x : x + x_sz] = prob[
                        -20:, :y_sz, :x_sz
                    ]
                    z_prev_flow[:, :, y : y + y_sz, x : x + x_sz] = flow[
                        :, -20:, :y_sz, :x_sz
                    ]
                else:
                    z_curr_prob[:, y : y + y_sz, x : x + x_sz] = prob[
                        :20, :y_sz, :x_sz
                    ]
                    z_curr_flow[:, :, y : y + y_sz, x : x + x_sz] = flow[
                        :, :20, :y_sz, :x_sz
                    ]
                    z_next_prob[:, y : y + y_sz, x : x + x_sz] = prob[
                        -20:, :y_sz, :x_sz
                    ]
                    z_next_flow[:, :, y : y + y_sz, x : x + x_sz] = flow[
                        :, -20:, :y_sz, :x_sz
                    ]

            if y != 0:
                ### not the first row, begin stitch the gap alone y ###
                for stitch_x in range(0, W, 1024):
                    stitch_x_sz = min(1024, W - stitch_x)

                    stitch_prob = np.concatenate(
                        (
                            y_prev_prob[
                                :, :, stitch_x : stitch_x + stitch_x_sz
                            ],
                            y_curr_prob[
                                :, :, stitch_x : stitch_x + stitch_x_sz
                            ],
                        ),
                        axis=-2,
                    )

                    if (stitch_prob < 0.5).all():
                        pass
                    else:
                        stitch_flow = np.concatenate(
                            (
                                y_prev_flow[
                                    :, :, :, stitch_x : stitch_x + stitch_x_sz
                                ],
                                y_curr_flow[
                                    :, :, :, stitch_x : stitch_x + stitch_x_sz
                                ],
                            ),
                            axis=-2,
                        )

                        stitch_mask, _ = compute_masks(
                            stitch_flow,
                            stitch_prob,
                            min_size=0,
                            flow_threshold=None,
                            cellprob_threshold=0.5,
                            do_3D=True,
                        )
                        org_mask = wholebrain[
                            z : z + z_sz,
                            y - 20 : y + 20,
                            stitch_x : stitch_x + stitch_x_sz,
                        ]
                        remap_mask = remap(org_mask, stitch_mask)
                        loadmask(
                            wholebrain,
                            remap_mask,
                            z,
                            z + z_sz,
                            y - 20,
                            y + 20,
                            stitch_x,
                            stitch_x + stitch_x_sz,
                        )

                y_prev_prob = y_next_prob.copy()
                y_prev_flow = y_next_flow.copy()

        if z != 0:
            ### not the first plane, begin stitch the gap alone z ###
            for stitch_y in range(0, H, 1024):
                stitch_y_sz = min(1024, H - stitch_y)
                for stitch_x in range(0, W, 1024):
                    stitch_x_sz = min(1024, W - stitch_x)
                    stitch_prob = np.concatenate(
                        (
                            z_prev_prob[
                                :,
                                stitch_y : stitch_y + stitch_y_sz,
                                stitch_x : stitch_x + stitch_x_sz,
                            ],
                            z_curr_prob[
                                :,
                                stitch_y : stitch_y + stitch_y_sz,
                                stitch_x : stitch_x + stitch_x_sz,
                            ],
                        ),
                        axis=-3,
                    )
                    if (stitch_prob < 0.5).all():
                        pass
                    else:
                        stitch_flow = np.concatenate(
                            (
                                z_prev_flow[
                                    :,
                                    :,
                                    stitch_y : stitch_y + stitch_y_sz,
                                    stitch_x : stitch_x + stitch_x_sz,
                                ],
                                z_curr_flow[
                                    :,
                                    :,
                                    stitch_y : stitch_y + stitch_y_sz,
                                    stitch_x : stitch_x + stitch_x_sz,
                                ],
                            ),
                            axis=-3,
                        )

                        stitch_mask, _ = compute_masks(
                            stitch_flow,
                            stitch_prob,
                            min_size=0,
                            flow_threshold=None,
                            cellprob_threshold=0.5,
                            do_3D=True,
                        )
                        org_mask = wholebrain[
                            z - 20 : z + 20,
                            stitch_y : stitch_y + stitch_y_sz,
                            stitch_x : stitch_x + stitch_x_sz,
                        ]
                        remap_mask = remap(org_mask, stitch_mask)
                        loadmask(
                            wholebrain,
                            remap_mask,
                            z - 20,
                            z + 20,
                            stitch_y,
                            stitch_y + stitch_y_sz,
                            stitch_x,
                            stitch_x + stitch_x_sz,
                        )

            z_prev_prob = z_next_prob.copy()
            z_prev_flow = z_next_flow.copy()

    print("All chunks written and dataset flushed successfully.")
