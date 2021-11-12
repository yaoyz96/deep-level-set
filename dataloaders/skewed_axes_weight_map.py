from sympy.geometry import Segment2D, Point2D, Line, intersection
from sympy.geometry.line import Line2D
from sympy.sets import EmptySet
import numpy as np
import random
from dataloaders.implementation import *
import cv2 as cv

def get_dist_from_line_segment(x1, c, x2, p, sig1, sig2):
    '''Given three colinear points, x1, c, x2 (geometric order), this program
    checks if point p and x1 are on the same side of a perpendicular to line segment x1x2 drawn at c.
    If p and x1 are on the same side of perpendicular line, the distance of p from line segment x1x2
    is divided by sig1 (for scaling) or by sig2 otherwise.
    Inputs:
        x1, c, x2, p: Np.array of size 2.
        sig1, sig2: scaling standard deviation
    Output:
        l1: Equivalent of l1 distance from the line segment x1x2 (scaled by sigma)'''
    assert isinstance_multiple([x1, x2, c], np.ndarray), "Either x1,x2 or c isn't a np.ndarray, check"
    assert x1.size==x2.size==c.size==2, "x1, x2, c must contain ony 2 elements"
    assert isinstance_multiple([sig1, sig2], float), "Either sig1 or sig2 is not of type float, check"
    assert not(sig1<0.001 or sig2<0.001), "sig1, sig2 is smaller than 0.001 (eps), check"

    x1, x2, c, p = [Point2D(i) for i in [x1, x2, c, p]]
    S  = Segment2D(x1,x2)
    L  = S.perpendicular_line(c)
    Sc = Segment2D(x1, p)
    l1 = S.distance(p).evalf()

    if isinstance(L.intersect(Sc), EmptySet):
        l1 = l1/sig1
    else:
        l1 = l1/sig2    
    return float(l1)

def get_R_T_from_points(x1, x2, c):
    '''This function returns a rotation (R), translation (T), inverse-rotation(Rinv) and inverse-translation (Tinv)
    matrices given two points (x1,x2) connected by a line and a point (c) which is the center of rotation
    NOTE: You can use these matrices as follows:
    1) To undo the rotation and translation deduced by looking at x1x2 line and c:
        [x,y,1] * Tinv * Rinv 
    2) R, T, Tinv matrices are constructed for right multiplication in this order:
        [x,y,1] * T * R * Tinv
    Inputs: 
        x1, x2, c - np.array of size 2
    Outputs:
        R,T,Rinv,Tinv - (3,3) np.array of rotation, translation, and inverse-translation matrices
        '''
    assert isinstance_multiple([x1,x2,c], np.ndarray), "Either x1,x2,or c isn't a np.ndarray, check"
    assert x1.size == x2.size==c.size==2, "x1,x2,c size must be 2, check"

    # Compute rotation wrt x-axis
    vec   = x2-x1
    denom = np.linalg.norm(vec)
    eps   = 0.00000001
    if denom > eps:
        theta = np.arccos(vec[1]/denom)
    else:
        theta = 0

    # Construct R, and T matrices
    R = np.identity(3)
    a = np.cos(theta)
    b = np.sin(theta)
    R[0:2,0:2] = np.array([[ a, b],[-b, a]])
    Rinv = R.T
    T = np.identity(3)
    T[2,:2] = c

    Tinv = np.identity(3)
    Tinv[2,:2] = -1*c
    
    return R, T, Rinv, Tinv

def isinstance_multiple(listObj, types):
    '''This function checks each object in listObj to be of type provided in types
    Inputs:
        listObj = list of objects to be typechecked
        types   = a type or a tuple of types
    Output:
        Bool representing the outcome of typechecking'''
    assert isinstance(listObj, list), "listObj must be a list"
    for i in listObj:
        if not isinstance(i, types):
            return False
    return True

def getPointOfIntersection(extreme_points):
    '''This function returns a unique point of intersection (if it exists)
    between four points in 2D plane.
    Input:
        extreme_points: (4,2) numpy array containing (x,y) coordinates of four points
    Output:
        intersection_point: A list containing [xint, yint]
    NOTE: This function errors out (ValueError) unless the intersection is a unique point. Implement
    error catching if this is undesired.'''

    assert isinstance(extreme_points, np.ndarray), "Exteme points should be passed as an ndarray"
    assert extreme_points.shape==(4,2), "Extreme point array shape must be (4,2)"

    # We try a random pairing based search. We make three attempts
    # to try unique pairings of fours points and look for the combination that gives
    # a unique intersection point.

    pairings           = [[0,1,2,3],[0,2,1,3],[0,3,1,2]]
    intersection_found = False
    i                  = 0
    pairs              = []

    while intersection_found is not True and i<3:
        pairs = pairings[i]
        x1 = Point2D( extreme_points[pairs[0],:] )
        x2 = Point2D( extreme_points[pairs[1],:] )
        x3 = Point2D( extreme_points[pairs[2],:] )
        x4 = Point2D( extreme_points[pairs[3],:] )

        # We use segment over line to ensure the intersection point lies within
        # the bounding box defined by the extreme points
        intersection_point = intersection(Segment2D(x1, x2), Segment2D(x3, x4))

        # Ensure that intersection point is a unique point and not a line or empty
        # if intersection_point == []:
        #     print('Can\'t find intersection point!!')
        if not intersection_point == []:
            if isinstance(intersection_point[0], Point2D):
                intersection_found = True
                xint, yint = intersection_point[0]
            # else:
            #     print('intersection point is not Point2D type!')
        i = i+1

    if intersection_found is not True:
        #raise ValueError("No intersection point was found for given extreme points using random pairing. Check")
        return 0, 0
    intersection_point = np.array([np.float128(xint.evalf()), np.float128(yint.evalf())])
    pairs              = np.array(pairs)
    return intersection_point, pairs

def getFourSigmas(extreme_points, intersection_point):
    '''Given 4 points and there point of intersection, this function
    approximates the measure of spread along these axes using their length
    from the intersection point
    Inputs:
        extreme_points - A (4,2) np.ndarray containing (x,y) locs of four points
        intersection_point - A np array of size 2
    Output:
        sigmas - a np array of size 4 containing the distances of 
                 four points from the center to approximate measure of spread'''

    assert isinstance(extreme_points, np.ndarray), "Exteme points should be passed as an ndarray"
    assert extreme_points.shape==(4,2), "Extreme point array shape must be (4,2)"
    assert isinstance(intersection_point, np.ndarray), "Exteme points should be passed as an ndarray"
    assert intersection_point.size==2, "Intersection point must be of size 2"

    '''This function returns the extreme points trimmed so they are 
    equidistant from the intersection point along their respective axes'''
    displacements = intersection_point - extreme_points.astype(np.float128)
    distances     = np.linalg.norm(displacements.astype(np.float128), axis=1)
    return distances

def get_theta_line_segment(x1, x2):
    '''This function returns the angle of vector x1--->x2 wrt x-axis in (0,pi) range
    Inputs: 
        x1, x2 - np.array of size 2
    Outputs:
        theta - in radians (0,pi) range
        '''
    assert isinstance_multiple([x1,x2], np.ndarray), "Either x1, or x2 isn't a np.ndarray, check"
    assert x1.size==x2.size==2, "x1,x2 size must be 2, check"

    # Compute rotation wrt x-axis
    vec   = x2.astype(np.float128)-x1.astype(np.float128)
    denom = np.linalg.norm(vec.astype(np.float128))
    eps   = 0.000000000000000000000000000000000001
    vec[0] = vec[0]+eps
    theta  = np.arctan(vec[1]/vec[0])
    
    # The step below ensures that rotation matrices are applied
    # such that there is only clockwise rotations of lines.
    # This fixes the relative position of extreme points before
    # and after rotation..
    if theta < 0:
        theta = theta+np.pi

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return theta, cos_theta, sin_theta

def undo_rot_trans_on_point(pt, theta, trans, cos_theta, sin_theta):
    '''This function undoes the rotation and translation on a point (pt).
    Input:
        pt - a np.array of size 2
        theta - a float representing the rotation to be undone in radians
        trans - a np.array of size 2 representing translation to be undone
    Output:
        pt - a np.array of size 2 calced as: (pt-c)*R -> R (rotmat) calced using -1*theta
    '''
    assert isinstance_multiple([theta, cos_theta, sin_theta], np.float128), "Theta, cos_theta, sin_theta must be in np.float128"
    assert isinstance_multiple([pt, trans], np.ndarray), "pt, trans must be of type np.ndarray"
    assert pt.dtype==trans.dtype==np.float128, "pt, trans dtype must be np.float128"
    
    # First undo translation
    pt      = pt-trans

    # Next, we apply rotation correction. We use the following property of rotation
    # matrix R_inverse = R_transpose. We further simply matrix multiplication to 
    # explicit operations as defined below for speedup.
    # We use the convention of applying rotations as a right multiplication matrix.
    # E.g. : [x y 1]*[R]
    pt1     = np.copy(pt)
    pt[0]   =    pt1[0]*cos_theta + pt1[1]*sin_theta
    pt[1]   = -1*pt1[0]*sin_theta + pt1[1]*cos_theta

    return pt

def get_dist_from_line_segment_fast(x1, c, x2, ptsx, ptsy, sig1, sig2, equal_sigma=True, scale=True):
    '''This function quickly computes distance of points in the 2D image from the line segment.
    Input:
        x1, c, x2  - a np.ndarray of size 2 with dtype=np.float128, they define a line segment x1---c---x2
        ptsx, ptsy - 2D (np.float128) np.ndarray of x and y coordinates for which we want the distance
        sig1, sig2 - scalar values in np.float128, these are 'measures of spread' along x1---c and c---x2 respectively
        equal_sigma- a flag to indicate if we want to use an equal (mean) sigma along x1--c and c--x2
        scale      - If this flag is set to True, we devide the distance by sig1/sig2 or their mean (if equal_sigma==True)
    Output:
        distance   - 2D (np.float128) np.ndarray containing distances of points in ptsx, ptsy from x1-----x2
    '''
    assert isinstance_multiple([sig1, sig2], np.float128), "sig1, sig2 must be in np.float128"
    assert isinstance_multiple([x1, c, x2, ptsx, ptsy], np.ndarray), "x1, c, x2, ptsx, ptsy must be of type np.ndarray"
    assert x1.dtype==x2.dtype==c.dtype==ptsx.dtype==ptsy.dtype==np.float128, "x1, c, x2, ptsx, ptsy dtype must be np.float128"
    
    # TEST FOR ORDER OF X1 AND X2 REVERSED IN THE FUNCTION BELOW
    theta, cos_theta, sin_theta = get_theta_line_segment(x1, x2)
    # theta_inv = -1*theta

    # Undo translation
    ptsx1 = ptsx-c[0]
    ptsy1 = ptsy-c[1]

    # Undo rotation
    ptsx =    ptsx1*cos_theta + ptsy1*sin_theta
    ptsy = -1*ptsx1*sin_theta + ptsy1*cos_theta

    # Undo translation and rotation for x1, x2:
    x1   = undo_rot_trans_on_point(x1, theta, c, cos_theta, sin_theta)
    x2   = undo_rot_trans_on_point(x2, theta, c, cos_theta, sin_theta)

    # Ensure x1 is to the right of c (origin) and x2 is to the left
    if x1[0]<x2[0]:
        x3 = np.copy(x1)  # holds x1
        x1 = np.copy(x2)  # holds point to the right of c
        x2 = np.copy(x3)  # holds point to the  left of c
        sig3 = sig1
        sig1 = sig2
        sig2 = sig3

    # figure out which points lie to the right of perpendicular drawn 
    # at c along x2-->x1 line segment
    right = ptsx > 0
    left  = ~right

    ## CALCULATE DISTANCE
    # Calculate distance of points from x1, x2, and the projection-distance
    x1c_dist    = np.sqrt((ptsx-x1[0])**2 + (ptsy)**2)
    x2c_dist    = np.sqrt((ptsx-x2[0])**2 + (ptsy)**2)
    proj_dist   = np.copy(np.abs(ptsy))
    # Figure out when each of the above three distances are applicable
    x1_mask     = ptsx > x1[0]
    x2_mask     = ptsx < x2[0]
    proj_mask   = ~(x1_mask | x2_mask)
    # Merge the distances using masks into a single distance
    # this is the distance of points from line segment! All vectorized, YAY!
    distance    = x1c_dist*x1_mask.astype(np.float128) + x2c_dist*x2_mask.astype(np.float128) + proj_dist*proj_mask.astype(np.float128)

    # Create scaling matrix:
    # sig_small = sig1
    # sig_flag  = False
    # if sig2<sig1:
    #     sig_small = sig2
    #     sig_flag  = True

    if equal_sigma is True:
        sig  = np.ones(ptsx.shape, dtype=np.float128)*(0.5*sig1+0.5*sig2)
        # if sig_flag is False:
        #     left_mask = 
        # sig  = np.ones(ptsx.shape, dtype=np.float128)*(0.5*sig1+0.5*sig2)
        # SIG1 = np.ones(ptsx.shape, dtype=np.float128)*sig1 * x1_mask
        # SIG2 = np.ones(ptsx.shape, dtype=np.float128)*sig2 * x2_mask
        # sigm = 0.5*sig1 + 0.5*sig2
        # proj_sig_r = np.ones(ptsx.shape, dtype=np.float128)*(np.abs(ptsx/x1[0])*(sig1-sig2)/2+sigm)*(right & proj_mask)
        # proj_sig_l = np.ones(ptsx.shape, dtype=np.float128)*(np.abs(ptsx/x2[0])*(sig2-sig1)/2+sigm)*( left & proj_mask)
        # sig  = proj_sig_l+proj_sig_r+SIG1+SIG2
        # pass
    else:
        SIG1 = np.ones(ptsx.shape, dtype=np.float128)*sig1
        SIG2 = np.ones(ptsx.shape, dtype=np.float128)*sig2
        sig  = SIG1*right + SIG2*left
    
    if scale: 
        distance    = distance / sig

    return distance # ndarray distance map

def compute_d1_d2_fast_skewed_axes(x, y, extreme_points, mahalonobis=1):
    '''This function quickly computes distance of points in the 2D image from the line segment.
    Input:
        x,y  - a np.array of containing the range of x and y coordinates
        extreme_points - a (4,2) np.ndarray of (x,y) coordinates of the extreme points (x1,x2,x3,x4)
        mahalonobis - a flag to indicate whether we want to use the true L2 distance (default) or approximate it using L1 distance
    Output:
        d1 - 2D (np.float128) np.ndarray containing chebyshev or chessboard distance of points from the line-segments (x1--x2, x3--x4)
        d2 - 2D (np.float128) np.ndarray containing L2 distance (or approximation using L1-distanc if mahalonobis=0)
    '''
    assert isinstance(extreme_points, np.ndarray), "Exteme points should be passed as an ndarray"
    assert extreme_points.shape==(4,2), "Extreme point array shape must be (4,2)"
    c, pairs         = getPointOfIntersection(extreme_points)
    if isinstance(c, int) and isinstance(pairs, int):
        if c==0 and pairs==0:
            return 0,0
    else:
        extreme_points   = extreme_points[pairs,:]
        [sig1, sig2, sig3, sig4] = getFourSigmas(extreme_points, c)
        x1 = extreme_points[0,:].astype(np.float128)
        x2 = extreme_points[1,:].astype(np.float128)
        x3 = extreme_points[2,:].astype(np.float128)
        x4 = extreme_points[3,:].astype(np.float128)
        c  = c.astype(np.float128)

        xmin, xmax = [np.min(x), np.max(x)]
        ymin, ymax = [np.min(y), np.max(y)]
        ptsx, ptsy = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
        ptsx = ptsx.astype(np.float128)
        ptsy = ptsy.astype(np.float128)

        d11 = get_dist_from_line_segment_fast(x1, c, x2, ptsx, ptsy, sig1, sig2, equal_sigma=True)
        d12 = get_dist_from_line_segment_fast(x3, c, x4, ptsx, ptsy, sig3, sig4, equal_sigma=True)
        d1  = np.minimum(np.abs(d11), np.abs(d12))

        if mahalonobis is True:
            d2  = np.sqrt(d11 **2 + d12 **2)
        else:
            d2  = np.abs(d11)  + np.abs(d12)

        return d1, d2 # ndarray distance map


def generate_mvL1L2_image_skewed_axes(mask, extreme_points=None, FULL_IMAGE_WEIGHTS=1, d2_THRESH=None, tau=1):
    '''This function generates a weight map given a binary mask. Operations:
        1) Extract 4 extreme points, pair them into intersecting line segments.
        2) Figure out the length, angles of the line segments.
        3) Assume these line segments form an coordinate-axes system, with intersection point at the origin.
        4) Generate a weight map such that weight decreases away from center and line segments, but equal to 1 all along the line segments.
    Input:
        mask  - a 2D np.ndarray of dtype=bool containing binary mask (1 for foreground)
        FULL_IMAGE_WEIGHTS -  A flag to check whether we compute on full image (default) or only on bounding box around extreme points
        d2_THRESH - A threshold to set 0 outside an ellipse given by mahalonobis distance, 2.5 to 4 is a good range
    Output:
        z   - 2D (np.float128) np.ndarray containing the weight map encoding distances as described above
        d1  - 2D (np.float128) np.ndarray containing the Chebyshev (Chessboard)-equivalent distance on skewed axes
        d2  - 2D (np.float128) np.ndarray containing the L2-equivalent distance on skewed axes
    '''
    assert isinstance(mask, np.ndarray), "2D mask must be numpy array"
    assert mask.dtype==np.bool, "Mask dtype must be np.bool"
    assert mask.ndim==2, "Mask must be 2D"
    if extreme_points is not None:
        assert isinstance(extreme_points, np.ndarray), "Extreme points must be np.ndarray"
        assert extreme_points.shape==(4,2), "Extreme points must have shape=(4,2)"

    y, x = np.nonzero(mask)

    x = np.arange(mask.shape[1])
    y = np.arange(mask.shape[0])
    if extreme_points is None:
        extreme_points = make_lines(mask)
    d1, d2 = compute_d1_d2_fast_skewed_axes(x, y, extreme_points, mahalonobis=1)
    z = 1/(1+d1*d2*tau)
    if d2_THRESH:
        z = z*(d2<d2_THRESH)
    z1 = np.zeros(tuple(list(z.shape)+[3]), z.dtype)
    z1[:,:,:] = z[:,:,np.newaxis]
    z1 = z1*255.01
    z1 = z1.astype(np.uint8)
    # cv.imwrite("TempZ.png", z1)
    return z, d1, d2

def make_lines(mask, stdDevMultiplier=6, angle_perturb=False):
    '''This function returns end points of major and minor axes of an ellipse fit
    on the mask (input) foreground.
    Input:
        mask - 2d np.ndarray dtype=bool, shows a binary mask with foreground=True
        stdDevMultiplier - Scalar, controls the extent of the major/minor axes drawn onto image
        angle_perturb - a bool to decide whether or not to randomly rotate one of the axes by a small amount
    Output:
        pts - (4,2) np.ndarray, containing end points of major/minor axes '''
    assert isinstance(mask, np.ndarray), "Mask must be a numpy.ndarray type"
    assert mask.ndim == 2, "Mask must be 2d"
    assert mask.dtype==np.bool, "Mask dtype must be np.bool"

    y, x = np.nonzero(mask)
    xbar, ybar, cov = compute_cov_xyMean(x, y)
    evals, evecs = np.linalg.eig(cov)
   
    random_angle_limit = 15
    if angle_perturb==True:
        theta = np.random.uniform(-1*random_angle_limit, random_angle_limit)
        theta = theta*np.pi/180
        R     = np.zeros((2,2))
        R[0,0]=    np.cos(theta)
        R[0,1]= -1*np.sin(theta)
        R[1,0]=    np.sin(theta)
        R[1,1]=    np.cos(theta)
        evecsR = evecs.T.dot(R).T
        axis = np.random.randint(0, 2)
        evecs[:,axis] = evecsR[:,axis]

    mean = np.array([xbar, ybar])
    '''Make lines a length of 2 stddev.'''
    pts = np.empty((4,2))
    for i in range(len(evals)):
        std = np.sqrt(evals[i])
        vec = stdDevMultiplier * std * evecs[:,i] / np.hypot(*evecs[:,i])
        pts[i*2,   :] = mean-vec
        pts[i*2+1, :] = mean+vec        
    return pts

def extreme_points(mask, pert):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    inds_y, inds_x = np.where(mask > 0.5)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)), # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)), # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)), # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert)) # bottom
                     ])