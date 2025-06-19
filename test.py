import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
import pandas as pd
from src.dataloader import get_dataloaders
from src.models.stego_gan import stegoganmodel
from src.util.util import tensor2im, save_image as save_img_util, mkdir


def mae(img1, img2):
    return torch.mean(torch.abs(img1 - img2)).item()

def rmse(img1, img2):
    return torch.sqrt(F.mse_loss(img1, img2)).item()

def evaluate(model, dataloader, device, results_dir):
    model.eval()
    mkdir(results_dir)
    mkdir(os.path.join(results_dir, "images"))
    metrics = []

    for i, data in enumerate(tqdm(dataloader, desc="Evaluating")):
        model.set_input(data)
        with torch.no_grad():
            model.forward()

        real_ct = model.real_B
        fake_ct = model.fake_B

        mae_score = mae(real_ct, fake_ct)
        rmse_score = rmse(real_ct, fake_ct)

        # Save generated images
        visuals = model.get_current_visuals()
        img_tensor = visuals['fake_B']
        save_path = os.path.join(results_dir, "images", f"fake_ct_{i}.png")
        save_image(img_tensor, save_path, normalize=True)

        metrics.append({
            "Sample": i,
            "MAE": mae_score,
            "RMSE": rmse_score
        })

    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(results_dir, "evaluation_metrics.csv"), index=False)
    print("Saved evaluation metrics to CSV.")

def main():
    data_root = "data/processed"
    split_dir = "data/splits"
    results_dir = "results/test"
    batch_size = 1
    use_mask = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader = get_dataloaders(data_root, split_dir, batch_size=batch_size, use_mask=use_mask)
    model = stegoganmodel()
    model.setup(device)

    evaluate(model, test_loader, device, results_dir)

if __name__ == '__main__':
    main()
