#Superpixel dataset preparation
import os
import cv2
import numpy as np
from skimage.segmentation import slic
from util.data_def import *

# Find smallest rectangle
def find_rec(segments, label):

  cluster_pts = np.where(segments == label)

  if cluster_pts[0].size == 0:
    return None

  else:
    cluster_y = cluster_pts[0]
    cluster_x = cluster_pts[1]

    max_x = max(cluster_x)
    max_y = max(cluster_y)
    min_x = min(cluster_x)
    min_y = min(cluster_y)

    return min_x, max_x, min_y, max_y

# Dilate the rectangle by 3 pixels
def dilation(coordinates, segments):

  height, width = segments.shape[0], segments.shape[1]

  if coordinates == None:
    return None

  else:
    min_x, max_x, min_y, max_y = coordinates

    min_x = max(0, min_x - 3)
    min_y = max(0, min_y - 3)
    max_x = min(width, max_x + 3)
    max_y = min(height, max_y + 3)

    return min_x, max_x, min_y, max_y

# Assign label to the superpuxel
def seg_classes(seg_img_p):

  height, width, _ = seg_img_p.shape
  label_counts = np.zeros(14)

  for i in range(height):
    for j in range(width):
      rgb = seg_img_p[i,j]
      ID = rgb_2_label[tuple(rgb)]

      label_counts[ID+1] += 1

  label = np.argmax(label_counts)
  return label


# Generate a folder and txt file that store each segmentation images and labels
if not os.path.exists(save_train_txt_path):
  os.mkdir(save_train_txt_path)

# Generate a folder that store the segmentation images
if not os.path.exists(save_data_path):
  os.mkdir(save_data_path)
  
  
if os.path.exists('./datasets/train.txt'):
    os.remove('./datasets/train.txt')

num_clusters = 100
print(msrc_directory)
for filename in os.listdir(msrc_directory):

    if (".bmp" in filename) and ("_GT" not in filename):
      train_filename = filename
      print("------------------------------> Processing image ", train_filename, " <------------------------------")

      # Read sample image then convert from BGR to RGB
      img = cv2.imread(os.path.join(msrc_directory, train_filename))
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # Read corresponding ground truth data and convert them to RGB
      GT_img = cv2.imread(os.path.join(msrc_directory, os.path.join(msrc_directory, train_filename.replace(train_filename[-4], "_GT."))))
      GT_img_rgb = cv2.cvtColor(GT_img, cv2.COLOR_BGR2RGB)

      # Run SLIC superpixel
      segments = slic(img_rgb, n_segments=100, compactness=10)

      # Iterate through each superpixel
      for k in range(num_clusters):

        # Find the rectangle and delate it
        rec_coord = find_rec(segments, k)
        dil_rec = dilation(rec_coord, segments)

        if dil_rec == None:
          continue
        else:

          # Crop the rectangle from original image
          min_x, max_x, min_y, max_y = dil_rec
          img_p = img_rgb[min_y:max_y, min_x:max_x]

          # Resize to 224*224
          img_p = cv2.resize(img_p, (224, 224), interpolation=cv2.INTER_AREA)

          # Cropp the ground truth segmentation image
          GT_img_p = GT_img_rgb[min_y:max_y, min_x:max_x]

          # Assign the label
          label = seg_classes(GT_img_p)

          # Save .npy file in designed path
          save_name = train_filename.replace(".bmp", "_" + str(k) + ".npy")
          save_name_path = save_data_path + save_name
          np.save(save_name_path, img_p)

          # Save .txt file in designed path
          f = open(save_train_txt_path + "/train.txt", "a+")
          f.write(save_name + "\t" + str(label) + "\n")
