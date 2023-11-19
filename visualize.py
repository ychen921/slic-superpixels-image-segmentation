import cv2
import numpy as np
import torch
from model import SegmentationNN
from torchvision import transforms
from skimage.segmentation import slic
from util.img_show import plot_image
from util.data_preprocessing import find_rec, dilation
import matplotlib.pyplot as plt
from util.data_def import *

transform = transforms.Compose([transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])



device = ("cuda")
model = SegmentationNN().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Visualize test segmentation map and ground truth segmentation map

org_img = cv2.imread(msrc_directory + "/" + "4_4_s.bmp")
seg_img = cv2.imread(msrc_directory + "/" + "4_4_s_GT.bmp")

org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)

plot_image(org_img, "Sample")
plot_image(seg_img, "Segmentation_Sample")

superpixel = slic(org_img, n_segments=100, compactness=10)

test_seg_map = np.zeros(org_img.shape, dtype=np.float32)


transform = transforms.Compose([transforms.ToTensor(),
                  transforms.Resize((224, 224)),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

for k in range(100):
  # Find the rectangle and delate it
  rec_coord = find_rec(superpixel, k)
  dil_rec = dilation(rec_coord, superpixel)

  if dil_rec == None:
    continue

  else:

    # Crop the rectangle from original image
    min_x, max_x, min_y, max_y = dil_rec
    img_p = org_img[min_y:max_y, min_x:max_x, :]

    # Resize to 224*224
    img_p = cv2.resize(img_p, (224, 224), interpolation=cv2.INTER_AREA)

    img_p = transform(img_p).to(device).unsqueeze(0)
    output = model(img_p.float())

    _, pred = torch.max(output, 1)
    ind = (superpixel == k)
    # test_seg_map[min_y:max_y, min_x:max_x, :] = label_2_rgb[int(pred.item()-1)]
    test_seg_map[ind] = label_2_rgb[int(pred.item()-1)]

print("hello")
plot_image(test_seg_map, "Segmentation_test")
plt.show()