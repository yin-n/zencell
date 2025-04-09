import models_vit
import torch

model = models_vit.__dict__[
    "vit_large_roipatch2x8x8_ctxpatch2x64x64_dep40_roi256_ctx1280_layerscale"
](out_chans=4)
model.half().eval().cuda()
model.load_state_dict(
    torch.load(
        "./checkpoint-4999.pth", map_location="cpu", weights_only=False
    )["model"]
)

print("Finish Load Model")
