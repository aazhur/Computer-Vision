import numpy as np
import matplotlib.pyplot as plt


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_3d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################
    
    B = np.reshape(points_2d, (points_2d.shape[0]*points_2d.shape[1],1))
    C11 = np.block([points_3d, np.tile(np.array([1,0,0,0,0]), (points_3d.shape[0],1))])
    C12 = np.block([points_3d, np.tile(np.array([1]), (points_3d.shape[0],1))])
    C12 = np.block([np.tile(np.array([0,0,0,0]), (points_3d.shape[0],1)), C12])
    C1 = np.block([C11, C12])
    C1 = np.reshape(C1, (2*C11.shape[0],C11.shape[1]))
    C2 = np.repeat(points_3d, 2, axis=0)
    C2 = (-1)*C2*np.repeat(B, 3, axis=1)
    A = np.block([C1, C2])
    
    M,_,_,_ = np.linalg.lstsq(A, B, rcond = None)
    M = np.vstack([M, [1]])
    M = np.reshape(M, (3,4))

    #raise NotImplementedError('`calculate_projection_matrix` function in ' +
    #    '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])
    
    Q = M[:,0:3]
    m4 = M[:,3]
    cc = np.dot((-1)*np.linalg.inv(Q), m4)

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

    #raise NotImplementedError('`calculate_camera_center` function in ' +
    #    '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################
    
    N = points_a.shape[0]
    m1 = np.mean(points_a, axis=0) 
    points_a_norm = points_a - np.tile(m1, (N,1))
    s1 = np.sqrt(np.sum(points_a_norm**2)/(2*N))
    pan = points_a_norm/s1; u1 = np.reshape(pan[:,0], (N,1)); v1 = np.reshape(pan[:,1], (N,1));
    Ta = np.asarray([[1/s1, 0, -m1[0]/s1],
    				 [0, 1/s1, -m1[1]/s1],
    				 [0, 0, 1]])
    
    #m = points_a[0,:]
    #m = np.append(m,1)
    #print(pan[0,:]) 
    #print(m)
    #print(np.matmul(Ta,m)) 				 
    
    m2 = np.mean(points_b, axis=0) 
    points_b_norm = points_b - np.tile(m2, (N,1))
    s2 = np.sqrt(np.sum(points_b_norm**2)/(2*N))
    pbn = points_b_norm/s2; u2 = np.reshape(pbn[:,0], (N,1)); v2 = np.reshape(pbn[:,1], (N,1));
    Tb_t = np.asarray([[1/s2, 0, 0],
    				 [0, 1/s2, 0],
    				 [-m2[0]/s2, -m2[1]/s2, 1]])
    
    #m = pbn[0,:]
    #m = np.append(m,1)
    #print(points_b[0,:]) 
    #print(m)
    #print(np.matmul(np.linalg.inv(np.transpose(Tb_t)),m)) 				 
    				 
    A = np.block([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, np.tile(np.array([1]), (N,1))])
    
    _, _, V = np.linalg.svd(A, full_matrices=False)
    f = V[V.shape[0]-1,:]
    F = np.reshape(f,(3,3))
        
    U, s, V = np.linalg.svd(F, full_matrices=False)
    S = np.diag(s)
    S[2,2] = 0
    F = np.matmul(U,np.matmul(S,V))
    F = np.matmul(Tb_t,np.matmul(F,Ta))
    
    #raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
    #    '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    N = matches_a.shape[0]
    
    thresh = 0.02
    c = 0
    s = 10
    S = np.asarray(np.zeros((N,1)))
    best_F = np.asarray(np.zeros((3,3)))
    best_S = S
    all_ind = np.linspace(0, (N-1), num = N, dtype = int)
    
    for i in range(100000):
    	#print((N-c)*100/N)
    	#k = all_ind; np.random.shuffle(k);
    	#ind = k[:s]
    	ind = np.random.randint(N, size=s)
    	
    	f = estimate_fundamental_matrix(matches_a[ind, :], matches_b[ind, :])
    	
    	u1 = np.reshape(matches_a[all_ind,0], (N,1)); v1 = np.reshape(matches_a[all_ind,1], (N,1));
    	u2 = np.reshape(matches_b[all_ind,0], (N,1)); v2 = np.reshape(matches_b[all_ind,1], (N,1));
    	A = np.block([f[0][0]*u1*u2, f[0][1]*v1*u2, f[0][2]*u2, f[1][0]*u1*v2, f[1][1]*v1*v2, f[1][2]*v2, f[2][0]*u1, f[2][1]*v1, np.tile(np.array([f[2][2]]), (N,1))])
    	S = np.sum(A, axis = 1)
    	S = np.absolute(S)
    	C = (S <= thresh)   	
    	if np.sum(C) > c:
    		c = np.sum(C)
    		best_F = f
    		best_S = S
    
    ind = np.argsort(best_S)
    
    inliers_a = matches_a[ind[:100], :]
    inliers_b = matches_b[ind[:100], :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    #raise NotImplementedError('`ransac_fundamental_matrix` function in ' +
    #    '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b
    