import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from logger import logger
from utils import get_model
from tqdm import tqdm

import torch
from torchvision.transforms import Compose, Normalize, ToTensor

import hydra
from hydra.utils import instantiate

_DEEPLABV3_DIR = Path(__file__).absolute().parents[0]
experiment_base_dir = _DEEPLABV3_DIR / "experiments"

def calculate_gray_white_ratio(mask):
    # Calculate the ratio of gray and white pixels in the mask
    gray_pixels = torch.sum(mask == 1).item()
    white_pixels = torch.sum(mask == 2).item()
    # total_pixels = mask.numel()
    total_pixels = gray_pixels + white_pixels
    gray_ratio = gray_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    return gray_ratio

def write_ratio_on_image(image, cloud_fraction):
    # Create a copy of the image to draw on
    image_with_text = image.copy()

    # Create an ImageDraw object to draw text
    draw = ImageDraw.Draw(image_with_text)

    # Define the font size and font color
    font_size = 30
    font_color = (255, 255, 255)  # White color

    # Define the text to display
    text = f"Cloud_fraction: {cloud_fraction:.2f}"

    # Get the dimensions of the image
    image_width, image_height = image.size

    # Get the size of the text to be displayed
    text_width, text_height = draw.textsize(text)

    # Calculate the position to center the text at the top of the image
    text_x = (image_width - text_width) // 2
    text_y = 10  # A small margin from the top

    # Draw the text on the image
    draw.text((text_x, text_y), text, font=None, fill=font_color)

    return image_with_text

@hydra.main(version_base=None, config_path=str(_DEEPLABV3_DIR), config_name="config")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.experiment_name = (
        cfg.experiment_name if cfg.experiment_name else datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    experiments_dir = experiment_base_dir / cfg.experiment_name

    cloud_dir = "../../data/sirta/"
    save_path = f"{str(experiments_dir)}/cloudfraction/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_paths = [cloud_dir + image for image in os.listdir(cloud_dir)]

    # print(image_paths)

    to_tensor = Compose([ToTensor(), Normalize(0.5, 0.5)])

    model = instantiate(cfg.model_params.model)
    model = get_model(
        model,
        device,
        checkpoint_path=f"{str(experiments_dir)}/models/model_best.pth",
    )
    model.to(device)
    logger.info("Calculating...\n")
    # Evaluation on test set
    model.eval()

    for path in tqdm(image_paths):
        image = Image.open(path)

        # Calculate the center crop dimensions
        crop_size = 1200
        left = (image.width - crop_size) // 2
        top = (image.height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        # Perform the center crop
        cropped_image = image.crop((left, top, right, bottom))

        image = cropped_image.resize((450, 450))
        image_tensor = to_tensor(image).unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            output = model(image_tensor)["out"]
            _, predicted = torch.max(output, 1)

        # Convert the predicted tensor back to a PIL image
        predicted_image = Image.fromarray((predicted.squeeze().cpu().numpy() * 255/2).astype('uint8'))
        # Calculate the gray and white ratios in the predicted mask
        cloud_fraction = calculate_gray_white_ratio(predicted)

        # Concatenate the original image and the predicted image side by side
        concatenated_image = Image.new("RGB", (image.width + predicted_image.width, image.height))
        concatenated_image.paste(image, (0, 0))
        concatenated_image.paste(predicted_image, (image.width, 0))
        
        # Write the ratios on the image
        image_with_cf = write_ratio_on_image(concatenated_image, cloud_fraction)
        # Save the image
        # concatenated_image.save(f"{save_path}/prediction_{os.path.basename(path)}")
        image_with_cf.save(f"{save_path}/prediction_{os.path.basename(path)}")

if __name__ == "__main__":
    main()