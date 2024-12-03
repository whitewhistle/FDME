import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
from skimage.segmentation import slic
from skimage.color import rgb2ycbcr
from scipy import sparse
from collections import defaultdict
from skimage.segmentation import mark_boundaries
import heapq
from skimage import color
from scipy.signal import convolve2d

def convert_bgr_to_rgb(image):
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input. Provide a valid numpy array.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def convert_rgb_to_gray(image):
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input. Provide a valid numpy array.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image
def apply_canny_edge_detection(image, low_threshold=10, high_threshold=50):
    # Apply Canny edge detection
    edges = cv2.Canny(image, low_threshold, high_threshold) 
    return edges
def g1x(x, y, s1):
    """
     Computes the gradient of a Gaussian filter in the x-direction.

    Parameters:
    x:ndarray
    The x-coordinates of the grid.
    y:ndarray
    The y-coordinates of the grid.
    s1:float
    Standard deviation of the Gaussian function.

    Returns:
    g:ndarray
    The gradient of the Gaussian filter in the x-direction.
    """
    # Calculate the square of the standard deviation
    s1sq = s1 ** 2

    # Below expression computes the gradient filter in the x-direction
    # The factor -x/(2π*s1^4) corresponds to the derivative of the Gaussian function with respect to x.
    # The exponential term exp(-(x^2+y^2)/(2*s1^2)) models the Gaussian distribution.
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(x, 2 * np.pi * s1sq ** 2),np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))
    return g

def g1y(x, y, s1):
    """
     Computes the gradient of a Gaussian filter in the y-direction.

    Parameters:
    x:ndarray
    The x-coordinates of the grid.
    y:ndarray
    The y-coordinates of the grid.
    s1:float
    Standard deviation of the Gaussian function.

    Returns:
    g:ndarray
    The gradient of the Gaussian filter in the y-direction.
    """
    # Calculate the square of the standard deviation
    s1sq = s1 ** 2

    # Below expression computes the gradient filter in the y-direction
    # The factor -y/(2π*s1^4) corresponds to the derivative of the Gaussian function with respect to y.
    # The exponential term exp(-(x^2+y^2)/(2*s1^2)) models the Gaussian distribution.
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(y, 2 * np.pi * s1sq ** 2),np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))
    return g

def estimate_sparse_blur(gimg, edge_map, std1, std2):
    half_window = 11
    m = half_window * 2 + 1
    a = np.arange(-half_window, half_window + 1)
    xmesh = np.tile(a, (m, 1))
    ymesh = xmesh.T

    f11 = g1x(xmesh, ymesh, std1) # It represents the gradient gaussian filter in x-direction with std as std1
    f12 = g1y(xmesh, ymesh, std1) # It repreents the gradient gaussian filter in y-direction with std as std1

    f21 = g1x(xmesh, ymesh, std2) # It represents the gradient gaussian filter in x-direction with std as std2
    f22 = g1y(xmesh, ymesh, std2) # It repreents the gradient gaussian filter in y-direction with std as std2

    #This Convolves the image with obtained gradient filter in x-direction and its also using nearest neighbourhood padding
    gimx1 = scipy.ndimage.convolve(gimg, f11, mode='nearest')
    
    #This Convolves the image with obtained gradient filter in y-direction and its also using nearest neighbourhood padding
    gimy1 = scipy.ndimage.convolve(gimg, f12, mode='nearest')
    
    #This computes gradient magnitude of pixel at(x,y) combining both the gradient in x and y direction using the filter with std as std1
    mg1 = np.sqrt(gimx1 ** 2 + gimy1 ** 2)
    
    #This Convolves the image with obtained gradient filter in x-direction and its also using nearest neighbourhood padding
    gimx2 = scipy.ndimage.convolve(gimg, f21, mode='nearest')
    
    #This Convolves the image with obtained gradient filter in y-direction and its also using nearest neighbourhood padding
    gimy2 = scipy.ndimage.convolve(gimg, f22, mode='nearest')
    
    #This computes gradient magnitude of pixel at(x,y) combining both the gradient obtained in x and y direction using the filter with std as std2
    mg2 = np.sqrt(gimx2 ** 2 + gimy2 ** 2)

    # Taking the ratio of gradients and adding a small value to avoid division by zero
    R = np.divide(mg1, mg2 + 1e-20) 
    R = np.multiply(R, edge_map > 0)  # Only consider edge locations

    # Estimate sparse blur values
    sparse_vals = np.divide(R ** 2 * (std1 ** 2) - (std2 ** 2), 1 - R ** 2 + 1e-10)
    sparse_vals[sparse_vals < 0] = 0

    sparse_bmap = np.sqrt(sparse_vals)
    sparse_bmap[np.isnan(sparse_bmap)] = 0
    sparse_bmap[sparse_bmap > 5] = 5  # Cap maximum value at 5

    return sparse_bmap,mg1,mg2


def slic_superpixel (image_rgb, N=200, compactness=10):
    # Apply SLIC to segment the image into superpixels
    superpixels = slic(image_rgb, n_segments=N, compactness=compactness, start_label=1)

    # Compute the mean color for each superpixel in the RGB color space
    superpixel_features = {}
    for i in range(1, np.max(superpixels) + 1):
        mask = superpixels == i
        superpixel_features[i] = np.mean(image_rgb[mask], axis=0)  # Compute mean for superpixel

    # Assign the mean color back to each superpixel
    superpixel_colored = np.zeros_like(image_rgb)
    for i in range(1, np.max(superpixels) + 1):
        mask = superpixels == i
        superpixel_colored[mask] = superpixel_features[i]  # Assign mean color to superpixel

    # Convert the superpixel_colored image back to RGB for visualization
    superpixel_colored_rgb = (superpixel_colored / np.max(superpixel_colored) * 255).astype(np.uint8)
    
    return superpixel_colored_rgb

def superpixel_defocus_propagation(image_rgb, n_segments=200, compactness=20, alpha=0.0001, sigma=4):
    superpixels = slic(image_rgb, n_segments=n_segments, compactness=compactness, start_label=1)
    print(superpixels)
    image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    
# Find adjacent superpixels by looking at each label of pixel to adjacent pixels 
    def find_adjacent_superpixels(labels):
        adjacency = defaultdict(set)
        for i in range(labels.shape[0] - 1):
            for j in range(labels.shape[1] - 1):
                current_sp = labels[i, j]
                neighbors = [labels[i + 1, j], labels[i, j + 1], 
                           labels[i + 1, j + 1], labels[i + 1, j - 1]]
                for neighbor in neighbors:
                    if neighbor != current_sp:
                        adjacency[current_sp].add(neighbor)
                        adjacency[neighbor].add(current_sp)
        return adjacency
    
    # Calculate mean color for each superpixel
    superpixel_features = {}
    for i in range(1, np.max(superpixels) + 1):
        mask = superpixels == i
        superpixel_features[i] = np.mean(image_ycrcb[mask], axis=0)
    
    # Get adjacency relationships
    adjacency = find_adjacent_superpixels(superpixels)
    
    # Initialize and compute weight matrix based on adjacency
    num_superpixels = np.max(superpixels)
    
    # Create sparse matrix entries
    rows = []
    cols = []
    weights = []
    
    for i in range(1, num_superpixels + 1):
        for j in adjacency[i]:
            # Compute Gaussian weight based on YCbCr feature similarity by taking mean YCbCr distance of each of these superpixels
            feature_i = superpixel_features[i]
            feature_j = superpixel_features[j]
            distance = np.linalg.norm(feature_i - feature_j)
            
            weight = np.exp(-distance ** 2 / (2 * sigma ** 2)) + 1e-36
            
            # Add symmetric entries
            rows.extend([i-1, j-1])  # -1 because sparse matrix is 0-based
            cols.extend([j-1, i-1])
            weights.extend([weight, weight])
    
    # Create sparse weight matrix that defines the Local relationship between each of these superpixels considering only the pixels that are adjacent
    W = sparse.csr_matrix((weights, (rows, cols)), 
                         shape=(num_superpixels, num_superpixels))
    
    # Compute degree matrix that deines the global relationship between each of these superpixels and needs to be used to calculate Transductive Affinity Matrix
    D = sparse.diags(W.sum(axis=1).A1)
    
    # Compute Transductive Affinity Matrix that defines similarity between each of these superpixels by using D and W 
    I = sparse.eye(num_superpixels)
    A = alpha * sparse.linalg.inv(D - (1-alpha)*W) @ I #Parameter alpha here denotes the priority given to D(global relationship) to W(Local Relationship)

    # The below part of code assign defocus values to each of the superpixels by taking mean values across the pixels in its egde by taking assigning it to median value of it 
    def propagate_defocus(sparse_defocus_map, seed_mask):
        # Initialize superpixel depth values
        val_sp = np.zeros(num_superpixels)
        for i in range(1, num_superpixels + 1):
            mask = (superpixels == i) & seed_mask
            if np.sum(mask) >= 2:  # Minimum number of seed points
                val_sp[i-1] = np.median(sparse_defocus_map[mask])
            else:
                val_sp[i-1] = -1
        
        # Set invalid depth columns to zero
        A_modified = A.copy()
        A_modified[:, val_sp == -1] = 0
        
        # Normalize rows to sum to 1
        
        row_sums = A_modified.sum(axis=1).A1

        # Avoid division by zero
        row_sums[row_sums == 0] = 1  
        A_hat = sparse.diags(1/row_sums) @ A_modified
        
# Propagate defocus estimated values using the Transductive Affinity matrix obtained above so through this we can take into consideration of how each of superpixels values are affected by adjacent superpixels
        new_val_sp = A_hat @ val_sp
        
        # Create final defocus map
        defocus_map = np.zeros_like(sparse_defocus_map)
        for i in range(1, num_superpixels + 1):
            defocus_map[superpixels == i] = new_val_sp[i-1]
        
        # Normalize to 0-1
        defocus_map = (defocus_map - np.min(defocus_map)) / (np.max(defocus_map) - np.min(defocus_map))
        return defocus_map
    
    return propagate_defocus, superpixels


def process_image_slic(image_rgb, sparse_defocus, seed_mask, n_segments, compactness):
    
    propagate_defocus, superpixels = superpixel_defocus_propagation(
        image_rgb, n_segments=n_segments, compactness=compactness
    )
    dense_defocus = propagate_defocus(sparse_defocus, seed_mask)
    
    return dense_defocus, superpixels
class Node:
    def __init__(self, index, label, dist):
        self.index = index
        self.label = label
        self.dist = dist
    
    def __lt__(self, other):
        return self.dist < other.dist

def push(heap, index, label, dist):
    node = Node(index, label, dist)
    heapq.heappush(heap, node)

def pop(heap):
    return heapq.heappop(heap)

def rgb_to_lab(img):
    return color.rgb2lab(img)

# For initializing seed we place them in a gris with roughly uniform spacing
def find_seeds(width, height, numk):
    size = width * height
    gridstep = int(np.sqrt(size / numk) + 0.5) 
    halfstep = gridstep // 2
    seeds = []
    for y in range(halfstep, height, gridstep):  
        for x in range(halfstep, width, gridstep):  
            seeds.append((x, y))  

    return seeds

def run_snic(image, numk, compactness):
    height, width, channels = image.shape
    sz = width * height
    lab_image = rgb_to_lab(image)
    
    # Initialize labels and other tracking variables
    labels = np.full((height, width), 0, dtype=int)  
    heap = []
    kx, ky, ksize = np.zeros(numk), np.zeros(numk), np.zeros(numk)
    kc = [np.zeros(numk) for _ in range(3)]  

    
    #We generate the seeds 
    seeds = find_seeds(width, height, numk)
    
    # We initialize the priority queue
    for k, (x, y) in enumerate(seeds):
        index = y * width + x  # Flattened pixel index
        push(heap, index, k, 0)  # Push the seed to the heap
    
    # Set compactness our hyperparameter
    M = compactness
    invwt = (M * M * numk) / sz  
    
    dx8 = [-1, 0, 1, 0, -1, 1, 1, -1]
    dy8 = [0, -1, 0, 1, -1, -1, 1, 1]
    
    pixel_count = 0  
     # We process each pixel based on distance and similarity
    while pixel_count < sz:
        node = pop(heap)
        ind, k, dist = node.index, node.label, node.dist
        x, y = ind % width, ind // width
        i = y * width + x
        
        if labels[y, x] == 0:  # If pixel hasn't been labeled yet
            labels[y, x] = k + 1  # Assign the superpixel label (start from 1)
            pixel_count += 1
            
            # Update the superpixel center (mean color and position)
            for c in range(3): 
                kc[c][k] += lab_image[y, x, c]
            kx[k] += x
            ky[k] += y
            ksize[k] += 1
            
            # We examine neighbors for 4-connectivity
            for p in range(4):  
                xx, yy = x + dx8[p], y + dy8[p]
                if 0 <= xx < width and 0 <= yy < height:  # Ensure within bounds
                    ii = yy * width + xx  # Flattened pixel index
                    if labels[yy, xx] == 0:  # If neighbor is unprocessed
                        # We compute color distance and spatial distance
                        colordist = sum([(kc[c][k] - lab_image[yy, xx, c] * ksize[k]) ** 2 for c in range(3)])
                        xdiff = kx[k] - xx * ksize[k]
                        ydiff = ky[k] - yy * ksize[k]
                        xydist = xdiff * xdiff + ydiff * ydiff
                        
                        # we combine color and spatial distance (compactness factor used)
                        slicdist = (colordist + xydist * invwt) / (ksize[k] ** 2)
                        
                        # we push the neighboring pixel into the heap for processing
                        push(heap, ii, k, slicdist)
    
    return labels    

def snic_superpixel_image (image, numk, compactness):
    # Generate superpixel labels using the SNIC algorithm
    labels = run_snic(image, numk, compactness)
    
    # Initialize an output image with the same shape as the input image
    output_image = np.zeros_like(image)
    
    # Loop over each unique label (superpixel)
    for label in np.unique(labels):
        # Create a mask for the current label
        mask = labels == label
        
        # Compute the mean color for the superpixel
        mean_color = image[mask].mean(axis=0)
        
        # Assign the mean color to the corresponding region in the output image
        output_image[mask] = mean_color

    return output_image

def superpixel_depth_propagation(image_rgb, n_segments=200, compactness=20, alpha=0.0001, sigma=4):
    superpixels = run_snic(image_rgb,n_segments,compactness)
    print(superpixels)
    image_ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    
# Find adjacent superpixels by looking at each label of pixel to adjacent pixels 
    def find_adjacent_superpixels(labels):
        adjacency = defaultdict(set)
        for i in range(labels.shape[0] - 1):
            for j in range(labels.shape[1] - 1):
                current_sp = labels[i, j]
                neighbors = [labels[i + 1, j], labels[i, j + 1], 
                           labels[i + 1, j + 1], labels[i + 1, j - 1]]
                for neighbor in neighbors:
                    if neighbor != current_sp:
                        adjacency[current_sp].add(neighbor)
                        adjacency[neighbor].add(current_sp)
        return adjacency
    
    # Calculate mean color for each superpixel
    superpixel_features = {}
    for i in range(1, np.max(superpixels) + 1):
        mask = superpixels == i
        superpixel_features[i] = np.mean(image_ycrcb[mask], axis=0)
    
    # Get adjacency relationships
    adjacency = find_adjacent_superpixels(superpixels)
    
    # Initialize and compute weight matrix based on adjacency
    num_superpixels = np.max(superpixels)
    
    # Create sparse matrix entries
    rows = []
    cols = []
    weights = []
    
    for i in range(1, num_superpixels + 1):
        for j in adjacency[i]:
            # Compute Gaussian weight based on YCbCr feature similarity by taking mean YCbCr distance of each of these superpixels
            feature_i = superpixel_features[i]
            feature_j = superpixel_features[j]
            distance = np.linalg.norm(feature_i - feature_j)
            
            weight = np.exp(-distance ** 2 / (2 * sigma ** 2)) + 1e-36
            
            # Add symmetric entries
            rows.extend([i-1, j-1])  # -1 because sparse matrix is 0-based
            cols.extend([j-1, i-1])
            weights.extend([weight, weight])
    
    # Create sparse weight matrix that defines the Local relationship between each of these superpixels considering only the pixels that are adjacent
    W = sparse.csr_matrix((weights, (rows, cols)), 
                         shape=(num_superpixels, num_superpixels))
    
    # Compute degree matrix that deines the global relationship between each of these superpixels and needs to be used to calculate Transductive Affinity Matrix
    D = sparse.diags(W.sum(axis=1).A1)
    
    # Compute Transductive Affinity Matrix that defines similarity between each of these superpixels by using D and W 
    I = sparse.eye(num_superpixels)
    A = alpha * sparse.linalg.inv(D - (1-alpha)*W) @ I #Parameter alpha here denotes the priority given to D(global relationship) to W(Local Relationship)

    # The below part of code assign defocus values to each of the superpixels by taking mean values across the pixels in its egde by taking assigning it to median value of it 
    def propagate_depth(sparse_depth_map, seed_mask):
        # Initialize superpixel depth values
        val_sp = np.zeros(num_superpixels)
        for i in range(1, num_superpixels + 1):
            mask = (superpixels == i) & seed_mask
            if np.sum(mask) >= 2:  # Minimum number of seed points
                val_sp[i-1] = np.median(sparse_depth_map[mask])
            else:
                val_sp[i-1] = -1
        
        # Set invalid depth columns to zero
        A_modified = A.copy()
        A_modified[:, val_sp == -1] = 0
        
        # Normalize rows to sum to 1
        
        row_sums = A_modified.sum(axis=1).A1

        # Avoid division by zero
        row_sums[row_sums == 0] = 1  
        A_hat = sparse.diags(1/row_sums) @ A_modified
        
# Propagate depth values using the Transductive Affinity matrix obtained above so through this we can take into consideration of how each of superpixels values are affected by adjacent superpixels
        new_val_sp = A_hat @ val_sp
        
        # Create final depth map
        depth_map = np.zeros_like(sparse_depth_map)
        for i in range(1, num_superpixels + 1):
            depth_map[superpixels == i] = new_val_sp[i-1]
        
        # Normalize to 0-1
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
        return depth_map
    
    return propagate_depth, superpixels


def process_image_snic(image_rgb, sparse_depth, seed_mask, n_segments, compactness):
    
    propagate_depth, superpixels = superpixel_depth_propagation(
        image_rgb, n_segments=n_segments, compactness=compactness
    )
    dense_depth = propagate_depth(sparse_depth, seed_mask)
    
    return dense_depth, superpixels




