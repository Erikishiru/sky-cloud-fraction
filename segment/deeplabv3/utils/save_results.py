import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor, ToPILImage
import os

def save_image(output_image, image_path):
    # if not isinstance(output_image, np.ndarray):
    #     if isinstance(output_image, torch.Tensor):  # get the data from a variable
    #         image_tensor = output_image.data
    #     # else:
    #     #     return output_image
    #     image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
    #     if image_numpy.shape[0] == 1:  # grayscale to RGB
    #         image_numpy = np.tile(image_numpy, (3, 1, 1))
    #     # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    # else:  # if it is a numpy array, do nothing
    #     image_numpy = output_image
    # output_image = output_image.data.cpu()
    if output_image.shape[0] == 3:
        image_numpy = output_image.permute(1, 2, 0).cpu().numpy()
        image_numpy = (image_numpy * 255).astype('uint8')
    else:
        image_numpy = output_image.cpu().numpy()
        image_numpy = (image_numpy * 255/2).astype('uint8')
    # image_numpy = (image_numpy * 255).astype('uint8')
    image_pil = Image.fromarray(image_numpy)
    # output_image = output_image.data.cpu()
    # img3 = Image.fromarray(t3.numpy())
    # image_pil = Image.fromarray(image_numpy)
    # if image_pil.mode != 'RGB':
    #     image_pil = image_pil.convert('RGB')
    # print(image_path)
    image_pil.save(image_path)

def save_image_tuples(images, labels, predictions, filenames, save_path):
    # print(images, labels, predictions)
    # print(images.shape, labels.shape, predictions.shape)
    # print(type(images), type(labels), type(predictions))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for idx in range(len(images)):
        denormalize = Normalize((-1, -1, -1), (2, 2, 2))
        # image_np = denormalize(images[idx]).squeeze().cpu().numpy()
        annotation_np = labels[idx].squeeze().cpu().numpy()
        prediction_np = predictions[idx].squeeze().cpu().numpy()

        # Convert NumPy arrays to PIL images
        # image_pil = Image.fromarray((image_np * 255).astype('uint8').transpose(1, 2, 0))  # Assuming image is in the format (C, H, W)
        to_pil_image = ToPILImage()
        image_pil = to_pil_image(denormalize(images[idx]))

        # Convert the reverted NumPy arrays to PIL images
        annotation_pil = Image.fromarray((annotation_np * 255/2).astype('uint8'))  # Assuming grayscale annotation
        prediction_pil = Image.fromarray((prediction_np * 255/2).astype('uint8'))  # Assuming grayscale prediction

        # Concatenate the images side by side
        concatenated_image = Image.new("RGB", (image_pil.width + annotation_pil.width + prediction_pil.width, image_pil.height))
        concatenated_image.paste(image_pil, (0, 0))
        concatenated_image.paste(annotation_pil, (image_pil.width, 0))
        concatenated_image.paste(prediction_pil, (image_pil.width + annotation_pil.width, 0))

        # Save the concatenated image with the original image filename
        filename = filenames[idx]
        output_path = os.path.join(save_path, f"concatenated_{filename}")
        concatenated_image.save(output_path)
        # save_image(images[idx], save_path + f'skyimage{idx}.png')
        # save_image(labels[idx], save_path + f'real_annotation{idx}.png')
        # save_image(predictions[idx], save_path + f'predicted_annotation{idx}.png')

def save_loss_plot(train_losses, val_losses, save_path):
    plot_file_path = os.fspath(save_path) + '/loss_plot.png'
    epochs = list(range(1, len(train_losses) + 1))
    # epochs = [tensor.cpu().numpy() for tensor in epochs]
    train_losses = [tensor.cpu().detach().numpy() for tensor in train_losses]
    val_losses = [tensor.cpu().detach().numpy() for tensor in val_losses]
    # print(type(epochs), type(train_losses), type(val_losses))
    # train_losses = train_losses.cpu().numpy()
    # val_losses = val_losses.cpu().numpy()
    if len(val_losses) > 0:
        # Plot the training and validation loss graph
        plt.plot(epochs, train_losses, label="Training Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.savefig(plot_file_path)
    else:
        plt.plot(epochs, train_losses, label="Training Loss")
        # plt.plot(epochs, val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()
        plt.savefig(plot_file_path)
