import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################
    
    cutoff = int(feature_width/4)
    gaussian = cv2.getGaussianKernel(cutoff*4,cutoff)
    filter = np.dot(gaussian, gaussian.T)
    
    dx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    dy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5) 
    magnitudes = np.hypot(dy.flatten(),dx.flatten())
    magnitudes = np.reshape(magnitudes,(image.shape[0],image.shape[1]))
    angles = np.degrees(np.arctan2(dy.flatten(),dx.flatten()))%360
    angles = np.reshape(angles, (image.shape[0],image.shape[1]))
    f = int(feature_width/2)
    p = int(feature_width/4)
    dlina = int(p**2)
    n_bins = 8
    step = int(360/n_bins)
    regions = list(range(0, 360+step, step))
    
    fv = []
    
    for k in range(len(x)):
    	angles_local = angles[int(y[k])-f:int(y[k])+f,int(x[k])-f:int(x[k])+f]
    	mag_local = magnitudes[int(y[k])-f:int(y[k])+f,int(x[k])-f:int(x[k])+f]
    	mag_local = filter*mag_local
    	#main_angle,_ = np.histogram(angles_local.flatten(),bins = regions, weights = mag_local.flatten())
    	#main_angle = np.argmax(main_angle)
    	for i in range(p):
    		for j in range(p):
    			array1 = angles_local[i*p:(i+1)*p,j*p:(j+1)*p].flatten()
    			array2 = mag_local[i*p:(i+1)*p,j*p:(j+1)*p].flatten()	
    			ind = i + j*p
    			results,_ = np.histogram(array1,bins = regions, weights = array2)
    			fv.append(results)
    			
    
    fv = np.reshape(fv,(len(x),n_bins*dlina))
    norms = np.array([np.linalg.norm(feature) for feature in fv])
    nonzero = np.array(np.where(norms > 0))
    array = np.reshape(np.repeat(norms[nonzero],n_bins*dlina),fv[nonzero].shape)
    fv[nonzero] = fv[nonzero]/array	
    
    fv[fv > 0.2] = 0.2
    
    norms = np.array([np.linalg.norm(feature) for feature in fv])
    nonzero = np.array(np.where(norms > 0))
    array = np.reshape(np.repeat(norms[nonzero],n_bins*dlina),fv[nonzero].shape)
    fv[nonzero] = fv[nonzero]/array	    
    fv = fv**(0.3)		
    		
    plt.figure(); plt.imshow(fv);		
    	

    #raise NotImplementedError('`get_features` function in ' +
    #    '`student_sift.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
