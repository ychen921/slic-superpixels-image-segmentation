import math
import cv2
import numpy as np
from tqdm import tqdm
from img_show import superpixel_plot, rgb_segment

def init_centroid(img, steps):
  centers = []

  """
  Randomly generate centers coordinate with loops

  Inputs:
  - img: A 3 channels numpy array, shape of (h, w, 3)
  - S: Avg length of superpixels, could also regard as step distance of each superpixels
  """
  h, w, _ = img.shape

  for i in range(steps//2, h, steps):
    for j in range(steps//2, w, steps):
      center_x = j 
      center_y = i 

      centers.append([center_x, center_y])

  return centers

def gradient(img):

  h, w, d = img.shape

  # Gradient map
  grad = np.zeros((h,w))

  # Compute the gradient
  for i in range(1, h-1):
    for j in range(1, w-1):
      for k in range(d):
        dx = img[i+1, j, k] - img[i-1, j, k]
        dy = img[i, j+1, k] - img[i, j-1, k]
        grad[i,j] = np.sqrt(dx**2 + dy**2)

  return grad

def perturb2_lowG(img, grad, centers):

  h, w, _ = img.shape
  num_centers = len(centers) # Number of centers

  k = 5 # 5X5 kernal

  for i in range(num_centers):
    x = centers[i][0]
    y = centers[i][1]

    # define region of neighbors
    kernal = grad[y-k : y+k, x-k : x+k]

    min_gradient_idx = np.argwhere(kernal == np.min(kernal))[0]

    # Move the center to the lowest gradient of nXn neighbor
    idx_y = min_gradient_idx[0] + y
    idx_x = min_gradient_idx[1] + x

    centers[i] = [idx_x, idx_y]

  return centers

def cielab_distance(pt_xy, pt_pixel, cnt_xy, cnt_pixel, S, m):
    
    pt_x = pt_xy[0]
    pt_y = pt_xy[1]
    pt_l, pt_a, pt_b = pt_pixel

    cnt_x = cnt_xy[0]
    cnt_y = cnt_xy[1]
    c_l, c_a, c_b = cnt_pixel

    d_xy = math.sqrt((pt_x-cnt_x)**2 + (pt_y-cnt_y)**2)
    d_lab = math.sqrt((pt_l-c_l)**2 + (pt_a-c_a)**2 + (pt_b-c_b)**2)

    dist = d_lab + (m*d_xy/S)

    return dist

def update_centers(img, segmap):

  h, w, _ = img.shape
  new_centers = []
  num_clusters = int(np.max(segmap)+1)

  for i in range(num_clusters):
    cluster_pts = np.where(segmap == i)

    cluster_y = cluster_pts[0]
    cluster_x = cluster_pts[1]

    center_y = cluster_y.mean()
    center_x = cluster_x.mean()


    new_centers.append([int(center_x), int(center_y)])

  return new_centers

def SLIC(im, k):
    """
    Input arguments:
    im: image input
    k: number of cluster segments


    Compute
    S: As described in the paper
    m: As described in the paper (use the same value as in the paper)
    follow the algorithm..

    returns:
    segmap: 2D matrix where each value corresponds to the image pixel's cluster number
    """
    height, width, depth = im.shape

    dist_map = np.ones((height, width))*math.inf
    segmap = np.zeros((height, width))

    # Initialize Parameters
    N = height * width     # Image pixels
    S = int(math.sqrt(N/k))      # Avg distance length of each cluster centers
    m = 10             # Range between 1 and 20
    convg_num = 10

    # Convert BRG to Lab space
    img_lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float64)

    # Initialize Centers
    centers = init_centroid(img_lab, S)

    # Compute the Gradient
    grad = gradient(img_lab)

    # Move the centers to the lowest gradient
    centers = perturb2_lowG(img_lab, grad, centers)

    iter = 0
    while iter < convg_num:
      distances = np.full((height, width), np.inf)
      # Assigning pixels to cluster
      for label in range(len(centers)):
        for i in range(height):
          for j in range(width):

            curr_xy = [j,i]
            curr_pixel = img_lab[i,j]

            centriod_pixel = img_lab[centers[label][1], centers[label][0]]

            # Find pixels within area arount pixel center on the D_s
            D_s = cielab_distance(curr_xy, curr_pixel, centers[label], centriod_pixel, S, m)

            if D_s < distances[i,j]:
              distances[i, j] = D_s
              segmap[i,j] = label


      centers = update_centers(img_lab, segmap)

      iter += 1

    return segmap


im_list = ['./data/MSRC_ObjCategImageDatabase_v1/1_22_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/1_27_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/3_3_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/3_6_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/6_5_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/7_19_s.bmp']

def main():
    for img in im_list:
        im = cv2.imread(img)
        k = 25
        clusters = SLIC(im, k=k)
        _ = rgb_segment(clusters,n = k, title =  "naive clustering: Pixelwise class plot: Clusters: " + str(k),legend = False)
        superpixel_plot(im,clusters,title =  "naive clustering: Superpixel plot: Clusters: "+ str(k))
  
if __name__ == "__main__":
    main()