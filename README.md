# slic-superpixels-image-segmentation
This project was composed of SLIC superpixel and image segmentation

## Superpixels
A superpixel can be defined as a group of pixels that share similar characteristics (such as intensity, or distance). Superpixel algorithms have been widely applied to various tasks like Image Segmentation and Object detection. 

In the first part of the project, we implement Kmean and SLIC superpixel algorithms.

### Perform k-means on image pixels `(r, g, b, x, y)`
The k-means clustering algorithm is an unsupervised algorithm which, for some items and for some specified number of clusters represented by cluster centers, minimizes the distance between items and their associated cluster centers. It does so by iteratively assigning items to a cluster and recomputing the cluster center based on the assigned items.

We implement the pixel clustering function `Kmean_superpixel.py`. It takes input an image (shape = (n, m, 3)) and number of clusters. Each pixel should be represented by a vector with 3 values: (r, g, b, x, y)

### SLIC superpixel
SLIC (Simple Linear Iterative Clustering) algorithm generates superpixels by clustering pixels based on color similarity and proximity in the image plane. We implement SLIC algorithm `SLIC_superpixel.py` from scratch and the detail of the algorithm can be found [here](https://www.iro.umontreal.ca/~mignotte/IFT6150/Articles/SLIC_Superpixels.pdf)


