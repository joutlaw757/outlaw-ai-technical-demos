import argparse
from pathlib import Path
from PIL import Image
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def predict(path):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights).eval()
    image = Image.open(path).convert("RGB")
    tensor = weights.transforms()(image).unsqueeze(0)
    import torch
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
    values, indices = probabilities.topk(5)
    for score, index in zip(values.tolist(), indices.tolist()):
        print(f"{weights.meta['categories'][index]}: {score:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    predict(args.image)
