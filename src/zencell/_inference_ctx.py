import os

import napari
import numpy as np
import torch
import zarr
from app_model.backends.qt import QMenuItemAction
from qtpy import QtWidgets
from qtpy.QtWidgets import QWidget

from zencell.zencell_model import models_vit
from zencell.zencell_model.cellpose.dynamics import compute_masks


class InferQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self._viewer = viewer

        self.layout = QtWidgets.QVBoxLayout()

        # self.setLayout(QHBoxLayout())

        # 1. Whole brain (zarr) path.
        self.whole_brain_label = QtWidgets.QLabel("Whole Brain Path (zarr):")
        self.whole_brain_input = QtWidgets.QLineEdit()
        self.layout.addWidget(self.whole_brain_label)
        self.layout.addWidget(self.whole_brain_input)

        # 2. Reference and segment channels (comma-separated).
        self.channels_label = QtWidgets.QLabel(
            "Reference Channel, Segment Channel (comma-separated):"
        )
        self.channels_input = QtWidgets.QLineEdit()
        self.layout.addWidget(self.channels_label)
        self.layout.addWidget(self.channels_input)

        # 3. Whole brain location (z, y, x).
        self.location_label = QtWidgets.QLabel(
            "Whole Brain Location (z, y, x):"
        )
        self.location_layout = QtWidgets.QHBoxLayout()
        self.location_z = QtWidgets.QLineEdit()
        self.location_z.setPlaceholderText("z")
        self.location_y = QtWidgets.QLineEdit()
        self.location_y.setPlaceholderText("y")
        self.location_x = QtWidgets.QLineEdit()
        self.location_x.setPlaceholderText("x")
        self.location_layout.addWidget(self.location_z)
        self.location_layout.addWidget(self.location_y)
        self.location_layout.addWidget(self.location_x)
        self.layout.addWidget(self.location_label)
        self.layout.addLayout(self.location_layout)

        # 4. Shape to segment (z, y, x).
        self.shape_label = QtWidgets.QLabel("Shape to Segment (z, y, x):")
        self.shape_layout = QtWidgets.QHBoxLayout()
        self.shape_z = QtWidgets.QLineEdit()
        self.shape_z.setPlaceholderText("z")
        self.shape_y = QtWidgets.QLineEdit()
        self.shape_y.setPlaceholderText("y")
        self.shape_x = QtWidgets.QLineEdit()
        self.shape_x.setPlaceholderText("x")
        self.shape_layout.addWidget(self.shape_z)
        self.shape_layout.addWidget(self.shape_y)
        self.shape_layout.addWidget(self.shape_x)
        self.layout.addWidget(self.shape_label)
        self.layout.addLayout(self.shape_layout)

        # 5. Volume dimensions (zmax, ymax, xmax) input.
        self.dimensions_label = QtWidgets.QLabel(
            "Volume Dimensions (zmax, ymax, xmax):"
        )
        self.dimensions_layout = QtWidgets.QHBoxLayout()
        self.zmax_input = QtWidgets.QLineEdit()
        self.zmax_input.setPlaceholderText("zmax")
        self.ymax_input = QtWidgets.QLineEdit()
        self.ymax_input.setPlaceholderText("ymax")
        self.xmax_input = QtWidgets.QLineEdit()
        self.xmax_input.setPlaceholderText("xmax")
        self.dimensions_layout.addWidget(self.zmax_input)
        self.dimensions_layout.addWidget(self.ymax_input)
        self.dimensions_layout.addWidget(self.xmax_input)
        self.layout.addWidget(self.dimensions_label)
        self.layout.addLayout(self.dimensions_layout)

        # 6. Output directory with a browse button.
        self.output_dir_label = QtWidgets.QLabel("Output Directory:")
        self.output_dir_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_output_dir)
        output_dir_layout = QtWidgets.QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_input)
        output_dir_layout.addWidget(self.browse_button)
        self.layout.addWidget(self.output_dir_label)
        self.layout.addLayout(output_dir_layout)

        # 7. Model selection.
        self.model_label = QtWidgets.QLabel("checkpoint Path (.pth):")
        self.model_ckpt = QtWidgets.QLineEdit()
        self.layout.addWidget(self.model_label)
        self.layout.addWidget(self.model_ckpt)

        # Run Inference button.
        self.run_button = QtWidgets.QPushButton("Run Inference")
        self.run_button.clicked.connect(self.run_inference)
        # self.run_button.clicked.connect(self._on_click)
        self.layout.addWidget(self.run_button)

        self.setLayout(self.layout)

    def browse_output_dir(self):
        """Open a directory selection dialog and update the output directory field."""
        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_input.setText(dir_path)

    def run_inference(self):
        """Collect UI parameters and execute the backend inference routine."""
        # --- Gather inputs from UI ---
        zarr_path = self.whole_brain_input.text()
        channels_text = self.channels_input.text()
        try:
            ref_chn, sig_chn = [
                int(x.strip()) for x in channels_text.split(",")
            ]
        except Exception:
            print(
                "Error parsing channels. Please input two numbers separated by a comma."
            )
            return

        try:
            z0 = int(self.location_z.text())
            y0 = int(self.location_y.text())
            x0 = int(self.location_x.text())
        except Exception:
            print(
                "Error parsing whole brain location. Please input valid integers for z, y, and x."
            )
            return

        try:
            seg_z = int(self.shape_z.text())
            seg_y = int(self.shape_y.text())
            seg_x = int(self.shape_x.text())
        except Exception:
            print(
                "Error parsing segmentation shape. Please input valid integers for z, y, and x."
            )
            return

        try:
            zmax = int(self.zmax_input.text())
            ymax = int(self.ymax_input.text())
            xmax = int(self.xmax_input.text())
        except Exception:
            print(
                "Error parsing volume dimensions. Please input valid integers for zmax, ymax, and xmax."
            )
            return

        output_dir = self.output_dir_input.text()

        # --- Set backend parameters ---
        # For this example, the checkpoint path is hardcoded.
        ckpt = self.model_ckpt.text()
        model_name = "vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale"

        # For simplicity, we assume a single process execution.
        rank = 0
        world_size = 1
        torch.cuda.set_device(rank % torch.cuda.device_count())

        # Load the model from the models_vit module.
        try:
            model = models_vit.__dict__[model_name](out_chans=4)
        except Exception as e:
            print("Error loading model:", e)
            return
        model.half().eval().cuda()
        model.load_state_dict(
            torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
        )

        print(f"Volume dimensions: zmax={zmax}, ymax={ymax}, xmax={xmax}")

        # Use the segmentation shape provided by the user.
        z, y, x = seg_z, seg_y, seg_x

        # --- Compute model-specific patch sizes and paddings ---
        rz, ry, rx = model.patch_embed_roi.img_size
        cz, cy, cx = model.patch_embed_ctx.img_size
        az, ay, ax = model.patch_embed_ctx.patch_size

        print(f"Model ROI patch size (rz, ry, rx): ({rz}, {ry}, {rx})")
        print(f"Segmentation shape (z, y, x): ({z}, {y}, {x})")

        # Ensure that the segmentation shape is divisible by the ROI patch size.
        try:
            assert (
                z % rz == 0 and y % ry == 0 and x % rx == 0
            ), "Segmentation shape must be divisible by the ROI patch size."
        except AssertionError as e:
            print(e)
            return

        zpad_head = (cz - rz) // 2 // az * az
        ypad_head = (cy - ry) // 2 // ay * ay
        xpad_head = (cx - rx) // 2 // ax * ax
        zpad_tail = cz - rz - zpad_head
        ypad_tail = cy - ry - ypad_head
        xpad_tail = cx - rx - xpad_head

        def crop_with_pad(chn, z0, y0, x0):
            """Crop the desired volume with padding if necessary."""
            z_st = z0 - zpad_head - rz // 2
            z_ed = z0 + z + zpad_tail + rz // 2
            y_st = y0 - ypad_head - ry // 2
            y_ed = y0 + y + ypad_tail + ry // 2
            x_st = x0 - xpad_head - rx // 2
            x_ed = x0 + x + xpad_tail + rx // 2
            z_slice = slice(max(z_st, 0), min(z_ed, zmax))
            y_slice = slice(max(y_st, 0), min(y_ed, ymax))
            x_slice = slice(max(x_st, 0), min(x_ed, xmax))
            z_pad = (max(-z_st, 0), max(z_ed - zmax, 0))
            y_pad = (max(-y_st, 0), max(y_ed - ymax, 0))
            x_pad = (max(-x_st, 0), max(x_ed - xmax, 0))
            arr = zarr.open(zarr_path, mode="r")["0"][
                chn, z_slice, y_slice, x_slice
            ]
            if any(a > 0 or b > 0 for a, b in (z_pad, y_pad, x_pad)):
                arr = np.pad(arr, (z_pad, y_pad, x_pad))
            return arr

        # Crop the region for both channels.
        ref_arr = crop_with_pad(ref_chn, z0, y0, x0)
        sig_arr = crop_with_pad(sig_chn, z0, y0, x0)

        # --- Set up sliding window offsets for segmentation ---
        z_offs_list, y_offs_list, x_offs_list = np.meshgrid(
            np.arange(0, z + 1, rz // 2),
            np.arange(0, y + 1, ry // 2),
            np.arange(0, x + 1, rx // 2),
        )
        z_offs_list = z_offs_list.reshape(-1).tolist()
        y_offs_list = y_offs_list.reshape(-1).tolist()
        x_offs_list = x_offs_list.reshape(-1).tolist()

        # Prepare tensors for accumulating the outputs.
        cell_prob = torch.zeros([z + rz, y + ry, x + rx], device="cuda")
        cell_flow = torch.zeros([3, z + rz, y + ry, x + rx], device="cuda")
        weight = torch.zeros([z + rz, y + ry, x + rx], device="cuda")

        # Create a weight template for blending the patches.
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

        # --- Run inference over the sliding window ---
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
                input_vol = torch.stack([ref_slice, sig_slice], dim=0).float()
                # Normalize input.
                input_vol = input_vol - input_vol.mean(
                    dim=(1, 2, 3), keepdim=True
                )
                input_vol = input_vol / (
                    input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6
                )
                input_vol = input_vol.half()
                with torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.CUDNN_ATTENTION]
                ):
                    slice_pred = model(input_vol[None])[0]
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

        # --- Save outputs ---
        out_filename_prob = os.path.join(
            output_dir,
            f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_prob.npy",
        )
        out_filename_flow = os.path.join(
            output_dir,
            f"sig{sig_chn}_ref{ref_chn}_z{z0:04d}_y{y0:04d}_x{x0:04d}_cell_flow.npy",
        )
        np.save(out_filename_prob, cell_prob)
        np.save(out_filename_flow, cell_flow)

        print("Inference complete. Results saved to:")
        print(out_filename_prob)
        print(out_filename_flow)

        # Assuming ref_arr and sig_arr are torch tensors on GPU from crop_with_pad
        # Convert them to CPU numpy arrays
        ref_np = ref_arr.cpu().numpy()
        sig_np = sig_arr.cpu().numpy()

        z_crop, y_crop, x_crop = ref_np.shape
        trimmed_ref = ref_np[
            z_crop // 2 - z // 2 : z_crop // 2 + z // 2,
            y_crop // 2 - y // 2 : y_crop // 2 + y // 2,
            x_crop // 2 - x // 2 : x_crop // 2 + x // 2,
        ]
        trimmed_sig = sig_np[
            z_crop // 2 - z // 2 : z_crop // 2 + z // 2,
            y_crop // 2 - y // 2 : y_crop // 2 + y // 2,
            x_crop // 2 - x // 2 : x_crop // 2 + x // 2,
        ]

        # Stack them along a new axis so that the shape becomes (2, 40, 1024, 1024)
        input_image = np.stack([trimmed_ref, trimmed_sig], axis=0)
        cellmask = compute_masks(
            cell_flow,
            cell_prob,
            min_size=0,
            flow_threshold=None,
            cellprob_threshold=0.5,
            do_3D=True,
        )[0]
        print(f"Cell mask shape: {cellmask.shape}")
        print(f"Input image shape: {input_image.shape}")

        # Now you can display it in napari as a multichannel image:
        self._viewer.add_image(input_image, name="Input Image")

        self._viewer.add_labels(cellmask, name="Cell Mask")

        # for whole brain
        QMenuItemAction._cache.clear()

        # Open a completely fresh viewer window
        viewer_whole = napari.Viewer(show=True)

        z_whole = (z0 + zmax) // 8
        y_whole = (y0 + ymax) // 8
        x_whole = (x0 + xmax) // 8

        patch_size = 128
        img_whole = zarr.open(zarr_path, mode="r")["2"][sig_chn][
            z_whole - 5 : z_whole + 5
        ]

        viewer_whole.add_image(
            img_whole,
            name=f"Z-plane {z_whole}",
            colormap="green",
            contrast_limits=[0, 65535],
        )
        viewer_whole.add_shapes(
            [
                [y_whole - patch_size, x_whole - patch_size],
                [y_whole + patch_size, x_whole - patch_size],
            ],
            edge_width=2,
            edge_color="white",
            ndim=2,
            shape_type="line",
        )
        viewer_whole.add_shapes(
            [
                [y_whole - patch_size, x_whole - patch_size],
                [y_whole - patch_size, x_whole + patch_size],
            ],
            edge_width=2,
            edge_color="white",
            ndim=2,
            shape_type="line",
        )

        viewer_whole.add_shapes(
            [
                [y_whole + patch_size, x_whole + patch_size],
                [y_whole + patch_size, x_whole - patch_size],
            ],
            edge_width=2,
            edge_color="white",
            ndim=2,
            shape_type="line",
        )
        viewer_whole.add_shapes(
            [
                [y_whole + patch_size, x_whole + patch_size],
                [y_whole - patch_size, x_whole + patch_size],
            ],
            edge_width=2,
            edge_color="white",
            ndim=2,
            shape_type="line",
        )

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")
