import os
from PIL import Image
import shutil

gen_data_path = '../../generate/pytorch-CycleGAN-and-pix2pix/results/WSISEG_pix2pix/test_latest/images/'
orig_data_path = './WSISEG-Database-split'

def copy_and_rename_images():
    # Create train directories if they don't exist
    train_whole_sky_path = os.path.join(orig_data_path, 'whole-sky-images', 'train')
    train_annotation_path = os.path.join(orig_data_path, 'annotation', 'train')
    os.makedirs(train_whole_sky_path, exist_ok=True)
    os.makedirs(train_annotation_path, exist_ok=True)
    
    # Loop through gen_data_path and copy samplexx_real.png and corresponding samplexx_fake.png
    for filename in os.listdir(gen_data_path):
        if filename.endswith('_real.png'):
            sample_id = filename.split('_')[0]
            new_real_filename = sample_id + '.png'
            shutil.copy(os.path.join(gen_data_path, filename),
                        os.path.join(train_whole_sky_path, new_real_filename))

            # print(os.path.join(train_whole_sky_path, new_real_filename))            
            fake_filename = sample_id + '_fake.png'
            if os.path.isfile(os.path.join(gen_data_path, fake_filename)):
                new_fake_filename = sample_id + '.png'
                shutil.copy(os.path.join(gen_data_path, fake_filename),
                            os.path.join(train_annotation_path, new_fake_filename))
                # Convert the fake annotation image to grayscale
                img = Image.open(os.path.join(train_annotation_path, new_fake_filename)).convert('L')
                
                # Manipulate pixel values to set gray pixels to 100, black pixels to 0, and white pixels to 255
                img_data = img.getdata()
                new_img_data = []
                for pixel in img_data:
                    if pixel < 86:  # Black pixels (0)
                        new_img_data.append(0)
                    elif pixel >= 86 and pixel < 192:  # Gray pixels (100)
                        new_img_data.append(100)
                    else:  # White pixels (255)
                        new_img_data.append(255)
                img.putdata(new_img_data)
                
                img.save(os.path.join(train_annotation_path, new_fake_filename))
                # print(os.path.join(train_annotation_path, new_fake_filename))


def split_and_copy_orig_data():
    # Create train, test, and val directories if they don't exist
    train_whole_sky_path = os.path.join(orig_data_path, 'whole-sky-images', 'train')
    test_whole_sky_path = os.path.join(orig_data_path, 'whole-sky-images', 'test')
    val_whole_sky_path = os.path.join(orig_data_path, 'whole-sky-images', 'val')
    train_annotation_path = os.path.join(orig_data_path, 'annotation', 'train')
    test_annotation_path = os.path.join(orig_data_path, 'annotation', 'test')
    val_annotation_path = os.path.join(orig_data_path, 'annotation', 'val')
    
    os.makedirs(train_whole_sky_path, exist_ok=True)
    os.makedirs(test_whole_sky_path, exist_ok=True)
    os.makedirs(val_whole_sky_path, exist_ok=True)
    os.makedirs(train_annotation_path, exist_ok=True)
    os.makedirs(test_annotation_path, exist_ok=True)
    os.makedirs(val_annotation_path, exist_ok=True)
    
    # Modify the following percentage splits as needed (e.g., 80%, 10%, 10%)
    train_split_percentage = 0.8
    test_split_percentage = 0.1
    val_split_percentage = 0.1
    
    # Loop through the files in orig_data_path and split them into train, test, and val
    files = [filename for filename in os.listdir(os.path.join(orig_data_path, 'whole-sky-images')) if os.path.isfile(os.path.join(orig_data_path, 'whole-sky-images', filename))]
    num_files = len(files)
    train_end = int(num_files * train_split_percentage)
    test_end = train_end + int(num_files * test_split_percentage)
    for i, filename in enumerate(files):
        if i < train_end and os.path.isfile(os.path.join(orig_data_path, 'whole-sky-images', filename)):
            shutil.copy(os.path.join(orig_data_path, 'whole-sky-images', filename),
                        os.path.join(train_whole_sky_path, filename))
            shutil.copy(os.path.join(orig_data_path, 'annotation', filename),
                        os.path.join(train_annotation_path, filename))
            # print(os.path.join(train_whole_sky_path, filename))
            # print(os.path.join(train_annotation_path, filename))
        elif i < test_end and os.path.isfile(os.path.join(orig_data_path, 'whole-sky-images', filename)):
            shutil.copy(os.path.join(orig_data_path, 'whole-sky-images', filename),
                        os.path.join(test_whole_sky_path, filename))
            shutil.copy(os.path.join(orig_data_path, 'annotation', filename),
                        os.path.join(test_annotation_path, filename))
        elif os.path.isfile(os.path.join(orig_data_path, 'whole-sky-images', filename)):
            shutil.copy(os.path.join(orig_data_path, 'whole-sky-images', filename),
                        os.path.join(val_whole_sky_path, filename))
            shutil.copy(os.path.join(orig_data_path, 'annotation', filename),
                        os.path.join(val_annotation_path, filename))

# Call the functions to perform the tasks
copy_and_rename_images()
split_and_copy_orig_data()

# Print the number of files in each directory and subdirectory
def count_files(directory):
    num_files = 0
    for root, _, files in os.walk(directory):
        num_files += len(files)
    return num_files

train_whole_sky_path = os.path.join(orig_data_path, 'whole-sky-images', 'train')
train_annotation_path = os.path.join(orig_data_path, 'annotation', 'train')
test_whole_sky_path = os.path.join(orig_data_path, 'whole-sky-images', 'test')
test_annotation_path = os.path.join(orig_data_path, 'annotation', 'test')
val_whole_sky_path = os.path.join(orig_data_path, 'whole-sky-images', 'val')
val_annotation_path = os.path.join(orig_data_path, 'annotation', 'val')

print("Number of files in train_whole_sky_path:", count_files(train_whole_sky_path))
print("Number of files in train_annotation_path:", count_files(train_annotation_path))
print("Number of files in test_whole_sky_path:", count_files(test_whole_sky_path))
print("Number of files in test_annotation_path:", count_files(test_annotation_path))
print("Number of files in val_whole_sky_path:", count_files(val_whole_sky_path))
print("Number of files in val_annotation_path:", count_files(val_annotation_path))