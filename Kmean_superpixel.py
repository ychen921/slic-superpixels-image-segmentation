import cv2
import numpy as np
from sklearn.cluster import KMeans
from img_show import superpixel_plot, rgb_segment, plot_image


def cluster_rgbxy(im,k):
    """
    Given image im and asked for k clusters, return nXm size 2D array
    segmap[0,0] is the class of pixel im[0,0,:]
    """
    #segmap is nXm. Each value in the 2D array is the cluster assigned to that pixel

    n, m, c = im.shape

    # Convert r g, b image into r, g, b, x, y , 5 channels
    x_channel = np.zeros((n, m))
    y_channel = np.zeros((n, m))

    # Store x-y coordinate info in 2 channels
    for i in range(n):
      for j in range(m):
        x_channel[i][j] = j
        y_channel[i][j] = i

    # Merge original RGB channels and XY channels
    convert_img = np.dstack((im, x_channel, y_channel))
    convert_img = np.float32(convert_img.reshape(-1, 5))

    # Implement Kmean clustering algorithm
    img_clusters = KMeans(n_clusters=k).fit(convert_img)

    # Reshape the image clusters to segmap (nXm)
    segmap = img_clusters.labels_.reshape(n,m)

    return segmap

im_list = ['./data/MSRC_ObjCategImageDatabase_v1/1_22_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/1_27_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/3_3_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/3_6_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/6_5_s.bmp',
           './data/MSRC_ObjCategImageDatabase_v1/7_19_s.bmp']


def main():
    im = cv2.imread(im_list[0])

    for k in [5, 10, 25, 50, 150]:
        clusters = cluster_rgbxy(im,k)
        _ = rgb_segment(clusters,n = k, title =  "naive clustering: Pixelwise class plot: Clusters: " + str(k),legend = False)
        superpixel_plot(im,clusters,title =  "naive clustering: Superpixel plot: Clusters: "+ str(k))
  
if __name__ == "__main__":
    main()