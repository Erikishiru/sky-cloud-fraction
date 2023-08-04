import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import wasserstein_distance

def calculate_probability_distribution(images):
    flattened_images = (images.reshape(images.shape[0], -1) * 255).astype(np.uint8)
    
    # Count occurrences of each pixel value and calculate frequencies
    pixel_counts = np.bincount(flattened_images.ravel(), minlength=256)
    frequencies = pixel_counts / np.sum(pixel_counts)
    
    # Create a dictionary with pixel values as keys and frequencies as values
    probability_distribution = dict(enumerate(frequencies))
    return probability_distribution

    # # Flatten the images and normalize pixel values to [0, 1]
    # flattened_images = images.reshape(images.shape[0], -1) / 255.0
    # print(flattened_images.shape)
    # # Count occurrences and calculate frequencies
    # unique, counts = np.unique(flattened_images, return_counts=True)
    # print(len(unique), len(counts))
    # frequencies = counts / np.sum(counts)
    
    # # Create a dictionary with pixel values as keys and frequencies as values
    # probability_distribution = dict(zip(unique, frequencies))
    # return probability_distribution

def kl_divergence(p, q):
    epsilon = 1e-9  # A small epsilon value to avoid division by zero
    p_smoothed = p + epsilon
    q_smoothed = q + epsilon
    kl_div = np.sum(p * np.log(p / q))
    return kl_div

dataDir = '../../data/WSISEG-Database/'
IMAGE_SIZE = 256
dataType = np.float32

image_filelist = glob.glob(dataDir + 'whole-sky-images/*')
image_filelist = sorted(image_filelist)
input_images = np.array([np.array(Image.open(fname).resize((IMAGE_SIZE,IMAGE_SIZE))) for fname in image_filelist]).astype(dataType)

gen_img_dir = './results_sky/WSISEG256/img_distributions/'

min_kl_divergence_value = float('inf')
min_kl_epoch = None

min_emd_value = float('inf')
min_emd_epoch = None

for i in range(20):
    epoch = (i+1)*250
    gen_filelist = glob.glob(gen_img_dir + f'epoch{epoch}/*')
    gen_filelist = sorted(gen_filelist)
    generated_images = np.array([np.array(Image.open(fname).resize((IMAGE_SIZE,IMAGE_SIZE))) for fname in gen_filelist]).astype(dataType)
    # Assuming you have 'input_images' and 'generated_images' as NumPy arrays
    input_distribution = calculate_probability_distribution(input_images)
    generated_distribution = calculate_probability_distribution(generated_images)
    # Convert dictionaries to NumPy arrays for KL divergence calculation
    input_probs = np.array(list(input_distribution.values()))
    generated_probs = np.array(list(generated_distribution.values()))
    # Calculate KL divergence
    kl_divergence_value = kl_divergence(input_probs, generated_probs)
    print(f"Epoch {epoch}:")
    print(f"KL Divergence:", kl_divergence_value)
    emd_value = wasserstein_distance(np.arange(256), np.arange(256), input_probs, generated_probs)
    print("Earth Mover's Distance (EMD):", emd_value) # A smaller EMD value indicates a closer similarity between the two sets of images.

    # Update minimum KL divergence
    if kl_divergence_value < min_kl_divergence_value:
        min_kl_divergence_value = kl_divergence_value
        min_kl_epoch = epoch

    # Update minimum EMD
    if emd_value < min_emd_value:
        min_emd_value = emd_value
        min_emd_epoch = epoch   

# Print the epochs with minimum KL divergence and minimum EMD
print("Epoch with Minimum KL Divergence:", min_kl_epoch)
print("Minimum KL Divergence:", min_kl_divergence_value)
print("Epoch with Minimum Earth Mover's Distance (EMD):", min_emd_epoch)
print("Minimum Earth Mover's Distance (EMD):", min_emd_value)
    