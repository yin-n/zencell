import models_vit
import numpy as np
import torch


def load_model(checkpoint_path="./checkpoint-4999.pth"):
    """Load the vision transformer model"""
    model = models_vit.__dict__[
        "vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale"
    ](out_chans=4)
    model.half().eval().cuda()
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)[
            "model"
        ]
    )
    print("Finish Load Model")
    return model


def run_inference(model, input_data):
    """
    Run inference with the loaded model

    Args:
        model: The loaded vision transformer model
        input_data: Input numpy array with shape (B, 2, D, H, W)
                   where D=40, H=1280, W=1280

    Returns:
        Output numpy array with model predictions
    """
    # Convert numpy array to tensor if needed
    if isinstance(input_data, np.ndarray):
        ref_slice = input_data[0]
        sig_slice = input_data[1]
        ref_tensor = (
            torch.from_numpy(ref_slice)
            if isinstance(ref_slice, np.ndarray)
            else ref_slice
        )
        sig_tensor = (
            torch.from_numpy(sig_slice)
            if isinstance(sig_slice, np.ndarray)
            else sig_slice
        )
        input_vol = torch.stack([ref_tensor, sig_tensor], dim=0).float()

        input_vol = input_vol - input_vol.mean(dim=(1, 2, 3), keepdim=True)
        input_vol = input_vol / (
            input_vol.std(dim=(1, 2, 3), keepdim=True) + 1e-6
        )
        input_vol = input_vol.half()
        input_tensor = input_vol.cuda()

    else:
        input_tensor = input_data.cuda().half()

    # Ensure batch dimension
    if input_tensor.dim() == 4:
        input_tensor = input_tensor.unsqueeze(0)

    print(f"Input tensor shape: {input_tensor.shape}")
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)

    # Convert output to numpy
    output_np = output.cpu().float().numpy()

    return output_np


def main():
    # Load the model
    model = load_model(checkpoint_path="./checkpoint-4999.pth")

    # Create sample input data (replace with your actual data loading)
    # Expected shape: (B, 2, 40, 1280, 1280)
    sample_input = np.load("./sample.npy")
    print(
        f"Intput shape: {sample_input.shape}"
    )  # Should be (1, 4, 40, 256, 256)
    # Run inference

    output = run_inference(model, sample_input)

    # Print output shape
    print(f"Output shape: {output.shape}")  # Should be (1, 4, 40, 256, 256)

    # Save output if needed
    np.save("output.npy", output)


if __name__ == "__main__":
    main()
