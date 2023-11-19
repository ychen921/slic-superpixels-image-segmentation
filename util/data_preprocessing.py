import numpy as np

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