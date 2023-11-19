import os
import torch
import numpy as np
import torch.utils.data as data
import pandas as pd

# -------------
# Dataset class
# -------------
#
# ``torch.utils.data.Dataset`` is an abstract class representing a
# dataset.
# Your custom dataset should inherit ``Dataset`` and override the following
# methods:
#
# -  ``__len__`` so that ``len(dataset)`` returns the size of the dataset.
# -  ``__getitem__`` to support the indexing such that ``dataset[i]`` can
#    be used to get sp_i sample
#
# Let's create a dataset class for our superpixel dataset. We will
# read the csv in ``__init__`` but leave the reading of images to
# ``__getitem__``. This is memory efficient because all the images are not
# stored in the memory at once but read as required.
#
# Sample of our dataset will be a dict
# ``{'superpixel image': image, 'superpixel class': class}``. Our dataset will take an
# optional argument ``transform`` so that any required processing can be
# applied on the sample. Remember to resize the image using ``transform``.

class SegmentationData(data.Dataset):

    def __init__(self, file, img_dir, transform=None):
        self.img_labels  = self.get_labels(file)
        self.img_dir = img_dir
        self.transform = transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = np.load(img_path)

        image = np.clip(np.asarray(image, dtype=float)/255, 0, 1)
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
          image = self.transform(image)

        return (image, label)

    def get_labels(self, txt_file):

        # Get Label from text file
        labels = open(txt_file, encoding='utf8').read().split('\n')

        data = []
        # Read each line
        for line in labels:
          data.append(line.split("\t"))
        data = pd.DataFrame(data)
        return data[:-1]
''' 
# Dataset Path and saving path of the dataset
current_directory = 'c:/Users/steve/Desktop/CMSC828I/Hw1/data'
msrc_directory = current_directory + '/MSRC_ObjCategImageDatabase_v1'
save_data_path = current_directory + '/Train_dataset/'
save_train_txt_path = current_directory + '/text_file/'
txt_file =  save_train_txt_path + "/train.txt"
img_path = save_data_path

# Get the dataset
seg_data = SegmentationData(txt_file, img_path)
sample_file_name = seg_data.img_labels[0][0].replace(".npy", ".bmp")

# Read and plot original and ground truth images
org_img = cv2.imread(msrc_directory + "/" + "4_4_s.bmp") #"MSRC_ObjCategImageDatabase_v1/" + sample_file_name)
plot_image(org_img, "Original_Sample")

seg_img = cv2.imread(msrc_directory + "/" + "4_4_s_GT.bmp") #sample_file_name.replace(".bmp", "_GT.bmp"))
plot_image(seg_img, "Segmentation_Sample")


SLIC_sample = slic(org_img, n_segments=100, compactness=10)
superpixel_plot(org_img, SLIC_sample, title = "SLIC Segmentation")
plt.show()

# Plot first 10 superpixels
i = 0
count = 1
while i < math.inf:
  slic_seg, label = seg_data[i]
  file_name = str(seg_data.img_labels[0][i])

  if (file_name == "4_4_s_1.npy") or (file_name == "4_4_s_2.npy") or (file_name == "4_4_s_3.npy") or (file_name == "4_4_s_4.npy") or (file_name == "4_4_s_5.npy") or (file_name == "4_4_s_6.npy") or (file_name == "4_4_s_7.npy") or (file_name == "4_4_s_8.npy") or (file_name == "4_4_s_9.npy") or (file_name == "4_4_s_10.npy"):
    print(str(seg_data.img_labels[0][i]))
    plt.imshow(slic_seg)
    plt.show()
    count += 1
  i+=1

  if count == 11:
    break
'''  