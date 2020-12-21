import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    #raise NotImplementedError('`get_interest_points` function in ' +
    #'`student_harris.py` needs to be implemented')
	
    #Ix = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
    #Iy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)
    cutoff = 4
    gaussian = cv2.getGaussianKernel(cutoff*4+1,cutoff)
    
    Iy,Ix = np.gradient(image)
    Ixx = Ix**2
    Ixx = cv2.filter2D(Ixx,-1,gaussian)
    Ixy = Ix*Iy
    Ixy = cv2.filter2D(Ixy,-1,gaussian)
    Iyy = Iy**2
    Iyy = cv2.filter2D(Iyy,-1,gaussian)
    k = 0.05
    w = int(feature_width/2)

    R = (Ixx*Iyy - Ixy*Ixy) - k*((Ixx+Iyy)**2)
    R[R < 0] = 0
    R[:w,:] = 0; R[:,:w] = 0; R[image.shape[0]-w:,:] = 0; R[:,image.shape[1]-w:] = 0
    [y,x] = np.nonzero(R)
    f = R[y,x]
    x = [i for _,i in sorted(zip(f,x), reverse = True)]
    y = [i for _,i in sorted(zip(f,y), reverse = True)]
    f = sorted(f, reverse = True)
    f = f[:int(0.05*len(f))]
    x = np.asarray(x[:len(f)])
    y = np.asarray(y[:len(f)])


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################
    
    r = [len(f),np.linalg.norm([x[1]-x[0],y[1]-y[0]])]
    for k in range(2,len(f)):
    	v1 = x[:k-1] - x[k]
    	v2 = y[:k-1] - y[k]
    	distance = np.hypot(v1,v2)
    	r.append(distance.min())
    
    x = [one for _,_,one in sorted(zip(r,f,x), reverse = True)]
    y = [one for _,_,one in sorted(zip(r,f,y), reverse = True)]
    n = 1500
    x = np.asarray(x[:n])
    y = np.asarray(y[:n])
    
        
    print('done')
    #raise NotImplementedError('adaptive non-maximal suppression in ' +
    #'`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return x,y, confidences, scales, orientations


