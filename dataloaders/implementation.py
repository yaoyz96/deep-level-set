import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
# import SimpleITK as sitk
# from pydicom import read_file
import cv2 as cv

def createGCMask(bg, fg, pfg, pbg):
    mask      = np.zeros(bg.shape[:2], dtype=np.uint8)
    if bg is not None:
        mask[bg] = 0
    if fg is not None:
        mask[fg] = 1
    if pbg is not None:
        mask[pbg] = 2
    if pfg is not None:
        mask[pfg] = 3

    return mask

def normalize_wtMap(wtmap):
    '''This function normalizes a 2D array of floats to 0 and 1 range
    Attribs:
        wtmap = 2D np array of floats
    output:
        normalized weight map, 2d array fo floats with range 0-1'''
    assert isinstance(wtmap, np.ndarray), "Weight map input must be np.ndarray"
    assert wtmap.ndim ==2, "Weight map must be only 2D"
    assert wtmap.dtype.kind=='f', "Weight map dtype must be a version of np.float"

    eps = 0.000001
    wtmap = wtmap-np.min(wtmap)
    wtmapMax = np.max(wtmap)
    if wtmapMax < eps:
        raise ValueError("The weight map is improperly scaled, check")
    wtmap /= wtmapMax
    return wtmap

def make_lines(mask, stdDevMultiplier=3):  
    y, x = np.nonzero(mask)
    xbar, ybar, cov = compute_cov_xyMean(x, y)
    evals, evecs = np.linalg.eig(cov)

    mean = np.array([xbar, ybar])
    """Make lines a length of 2 stddev."""
    xy = np.empty((4,2))
    for i in range(len(evals)):
        std = np.sqrt(evals[i])
        vec = stdDevMultiplier * std * evecs[:,i] / np.hypot(*evecs[:,i])
        # x, y = mean-vec
        # xy[i*2,   :] = [y,x]
        # x, y = mean+vec
        # xy[i*2+1, :] = [y,x]        
        xy[i*2,   :] = mean-vec
        xy[i*2+1, :] = mean+vec        
    return xy

def getMajorMinorLines(mask, stdDevMultiplier=3):
    xy = make_lines(mask, stdDevMultiplier)
    xy = tuple(xy.astype(np.uint16))
    mask = np.zeros(mask.shape[:2], np.uint8)
    cv.line(mask, tuple(xy[0]), tuple(xy[1]), 255,  1)
    cv.line(mask, tuple(xy[2]), tuple(xy[3]), 255,  1)
    return mask>250

def createMaskColor(mask, color='r'):
    '''This program accepts a 2d MxN mask and optional color argument
    as an input and returns an MxNx3 image with color correctly applied.
    mask  - 2D (uint8, or any numpy float).
    alpha - can be a scalar (float or np.float64) between 0 and 1, or a 2D transparency mask (uint8 or any numpy float)
    '''
    # Check type and format of the mask
    assert type(mask)==np.ndarray, 'image needs to be a numpy ndarray'
    assert mask.ndim==2 or (mask.dtype==np.uint8 or mask.dtype.kind=='f'), \
        'Mask should 2d, type must be uint8 or numpy float'
    assert color in ['r', 'g', 'b', 'y', 'c', 'm', 'w', 'bl'], "Color type unknown, chose from: 'r', 'g', 'b', 'y', 'c', 'm', 'w', 'bl'"

    colorMask = np.zeros(tuple(list(mask.shape)+[3]), dtype=mask.dtype)

    if color in ['r', 'y', 'm', 'w']:
        colorMask[:,:,0] = mask
    if color in ['g', 'y', 'c', 'w']:
        colorMask[:,:,1] = mask
    if color in ['b', 'c', 'm', 'w']:
        colorMask[:,:,2] = mask
    return colorMask

def imageAlpha(image, alpha=None):
    ''' This program accepts an MxN image of 1 or 3 channels and optional alpha channel
    as an input and returns an MxNx4 image with alpha channel set.
    image - 2D or 3D image (uint8, or any numpy float).
    alpha - can be a scalar (float or np.float64) between 0 and 1, or a 2D transparency mask (uint8 or any numpy float)
    '''
    # Check type and format of the image
    assert type(image)==np.ndarray, 'image needs to be a numpy ndarray'
    assert image.ndim==2 or (image.ndim==3 and image.shape[-1]==3), \
        'Image should be either 2d or 3d, if 3d, the third dim can only contain 3 channels'
    assert image.dtype==np.uint8 or image.dtype.kind=='f', 'Image data type must be uint8 or float'
    
    # Check type and format of alpha
    if alpha is not None: # User provided alpha, check it
        assert isinstance(alpha, (float, np.ndarray)), 'alpha should either be a scalar or \
        a numpy ndarray of type uint8 or float' 
        if type(alpha)==float:
            # Error check and convert a float alpha to an alpha channel
            assert 0.0 <= alpha <= 1.0, 'alpha must be between 0 and 1'
            alpha = np.ones(image.shape[:2], dtype=np.float)*alpha
        else:
            # We just make sure alpha is good to go
            assert alpha.ndim==2 and (alpha.dtype==np.uint8 or alpha.dtype.kind=='f'), \
            'Non-scalar alpha should be 2D np array of type uint8 or float'
            assert alpha.shape[:2] == image.shape[:2], "alpha channel and image dims mismatch"

            if alpha.dtype.kind=='f':
                alpha = alpha.astype(np.float)
            else:
                alpha = alpha.astype(np.float)/255.0

    else: # No alpha specified, set it to 1
        alpha = np.ones(image.shape[:2], dtype=np.float)
        
    # Convert the image to MxNx4 image, set the 4th channel to alpha
    m, n = image.shape[:2]
    image_alpha = np.empty(shape=(m, n, 4), dtype=image.dtype)

    if image.ndim==2:
        image = image[:,:,np.newaxis]
    image_alpha[:,:,:3] = image
    image_alpha[:,:, 3] = alpha
    return image_alpha

def colorMaskWithAlpha(mask, transparency=0.4, color='r', normalize=1):
    '''This program accepts a 2d MxN mask, transparency and color arguments
    and returns an MxNx4 image with color and transparency correctly applied.
    Transparency is set according to grayscale mask values.
    mask  - 2D (uint8, or any numpy float).
    transparency - scalar (float or np.float64) between 0 and 1, default=0.4
    color - must be one of the ['r', 'g', 'b', 'y', 'c', 'm', 'w']. default='r'
    normalize - the alpha channel is normalized, default=true
    '''
    assert isinstance(mask, np.ndarray), 'Mask must be numpy ndarray'
    assert mask.dtype.kind=='f' or mask.dtype==np.uint8, 'Mask dtype must be uint8 or numpy float'
    assert mask.ndim==2, 'Mask must be 2d'
    assert isinstance(transparency, float), 'Transparency is not a float'
    assert color in ['r', 'g', 'b', 'y', 'c', 'm', 'w', 'bl'], "Color type unknown, chose from: 'r', 'g', 'b', 'y', 'c', 'm', 'w', 'bl"
    
    alpha = np.copy(mask)

    if alpha.dtype==np.uint8:
        alpha = alpha.astype(np.float)/255.0
    else:
        alpha = alpha.astype(np.float)
        
    if normalize:
        alpha = alpha-np.min(alpha)
        if np.abs(np.max(alpha)) < np.finfo(alpha.dtype).eps:
            #raise ValueError('Mask has improper scaling, normalizing failed')
            alpha = alpha # We don't scale, the mask is likely just zeros everywhere
        else:
            alpha = alpha/np.max(alpha)

    alpha = alpha*transparency
    mask  = createMaskColor(mask, color)
    return imageAlpha(mask, alpha)

def plotGCMask(ax, mask, transparency=0.1, color=None):
    assert isinstance(mask, np.ndarray), 'Mask must be numpy ndarray'
    assert mask.dtype==np.uint8, 'Mask dtype must be uint8'
    assert mask.ndim==2, 'Mask must be 2d'
    
    if color is not None:
        assert isinstance(color, list), 'Color must be specified as a length=4 list of color tags (string)'
        assert len(color)==4, '''Color, when specified, must contain \
            four color tags arranged as: [bg, fg, pfg, pbg] \
            and possible color tags are: 'r', 'g', 'b', 'y', 'c', 'm', 'w' '''
        assert type(color[0]==color[1]==color[2]==color[3]==str), "All elements of color list must be string type"
    else:
        color = ['r', 'g', 'r', 'g']

    bg  = mask==0
    fg  = mask==1
    pfg = mask==3
    pbg = mask==2

    ax.imshow(colorMaskWithAlpha( bg.astype(np.float), transparency=0.1, color=color[0], normalize=0))
    ax.imshow(colorMaskWithAlpha( fg.astype(np.float), transparency=0.1, color=color[1], normalize=0))
    ax.imshow(colorMaskWithAlpha(pbg.astype(np.float), transparency=0.1, color=color[2], normalize=0))
    ax.imshow(colorMaskWithAlpha(pfg.astype(np.float), transparency=0.1, color=color[3], normalize=0))
    return None

def generate_mvL1L2_image(mask, FULL_IMAGE_WEIGHTS=1, d2_THRESH=1.7):
    y, x = np.nonzero(mask)
    xbar, ybar, cov = compute_cov_xyMean(x, y)
    evals, evecs = np.linalg.eig(cov)

    x = np.arange(mask.shape[1])
    y = np.arange(mask.shape[0])

    d1, d2 = compute_d1_d2_fast(x, y, xbar, ybar, evals, evecs, mahalonobis=1)
    d22 = d2*(d2<3)
    z = 1/(1+d1*d22)
    z = z*(d2<d2_THRESH)
    return z, d1, d2

def compute_d1_d2_fast(x, y, xbar, ybar, evals, evecs, mahalonobis=1):
    xmin, xmax = [np.min(x), np.max(x)]
    ymin, ymax = [np.min(y), np.max(y)]
    ptsx, ptsy = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    ptsx = ptsx-xbar
    ptsy = ptsy-ybar

    R_inverse = evecs.T
    Rinv_00 = R_inverse[0,0]
    Rinv_01 = R_inverse[0,1]
    Rinv_10 = R_inverse[1,0]
    Rinv_11 = R_inverse[1,1]

    Del_inv_sqrt = evals**-0.5
    alpha = 1
    normalized_x = (ptsx*Rinv_00 + ptsy*Rinv_01)*Del_inv_sqrt[0]*alpha
    normalized_y = (ptsx*Rinv_10 + ptsy*Rinv_11)*Del_inv_sqrt[1]*alpha

    d1 = np.minimum(np.abs(normalized_x), np.abs(normalized_y))
    
    if mahalonobis==1:
        d2 = np.sqrt(normalized_x**2 + normalized_y**2)
    else:
        d2 = np.abs(normalized_x) + np.abs(normalized_y)

    return d1, d2

def generate_mvgauss_image(img, FULL_IMAGE_WEIGHTS=0, tau = 1):
    '''Given a binary image with white foreground, this program
    computes two primary eigenvectors and covariance matrix to
    evaluate the distribution of weights around the center of mass
    of the blob using the 2D gaussian formula. The code has been
    optimized for execution speed.'''
    y, x = np.nonzero(img)
    xbar, ybar, cov = compute_cov_xyMean(x, y) # Center of mass and covariance

    if FULL_IMAGE_WEIGHTS==1:
        x = np.arange(img.shape[1])
        y = np.arange(img.shape[0])

    # Calculate PDF using the 2D gaussian formula
    pdf = compute_mvgauss_fast(x, y, xbar, ybar, cov, tau)
    pdf = pdf/np.max(pdf) # Normalize the pdf

    # Preallocate an output weights image and assign the PDF
    wts_img = np.zeros(img.shape)
    wts_img[np.min(y):np.max(y)+1, np.min(x):np.max(x)+1] = pdf

    return wts_img

def compute_cov_xyMean(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    x = x-xbar
    y = y-ybar
    coords = np.vstack([x, y]) # Change for better performance
    cov = np.cov(coords)       # Compute covariance matrix
    return xbar, ybar, cov

def compute_mvgauss_fast(x, y, xbar, ybar, cov, tau=1):
    # Set tau to less than 1 to compress the bell shape and vice-versa
   
    # Precalculate a meshgrid for fast computation of the function on the grid
    xmin, xmax = [np.min(x), np.max(x)]
    ymin, ymax = [np.min(y), np.max(y)]
    ptsx, ptsy = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    ptsx       = ptsx-xbar
    ptsy       = ptsy-ybar

    # Precalculate the inverse of the covariance matrix for quick computation
    # of the PDF, extract the unique elements from the 2by2 covariance matrix
    inv_cov    = np.linalg.inv(cov)
    inv_cov_00 = inv_cov[0,0]
    inv_cov_01 = inv_cov[0,1]
    inv_cov_11 = inv_cov[1,1]

    # The matrix multiplication has been decomposed for quick calculation
    pdf_exp    = np.exp(-0.5*(inv_cov_00*ptsx**2 + 2*inv_cov_01*ptsx*ptsy + inv_cov_11*ptsy**2)/tau)
    cov_meas   = np.linalg.det(cov)**-0.5
    pdf_const  = 1*cov_meas/(2*np.pi)
    pdf        = pdf_const* pdf_exp
   
    return pdf

def checkIfDir(*paths):
    '''This function uses os.path.join to conflate the 
    paths provided as input argument, and uses os.path.isdir
    to check if the conflated path is a directory'''
    return os.path.isdir(os.path.join(*paths))

def getDirectoriesInPath(path):
    '''This function obtains a file list at path, removes all those
    that aren't a directory, and returns the remaining items sorted'''
    fileList = os.listdir(path)
    fileList = [i for i in fileList if checkIfDir(path,i) is True]
    fileList.sort()
    return fileList

def loadDicomSeries(dirName):
    listDcmFiles = os.listdir(dirName)
    listDcmFiles = [i for i in listDcmFiles if i.endswith('.dcm')]
    listDcmFiles.sort()
    listDcmFiles = [os.path.join(dirName, i) for i in listDcmFiles]

    # Figure out the image props:
    img1 = read_file(listDcmFiles[0])
    imgSize = [img1.Rows, img1.Columns, len(listDcmFiles)]
    imgSize = tuple([int(i) for i in imgSize])

    # Load spacing values (in mm)
    if hasattr(img1, 'SliceThickness'):
        sliceThickness = img1.SliceThickness
    else:
        sliceThickness = img1.PixelSpacing[0]*1.4

    imgPixelSpacing = [float(img1.PixelSpacing[0]), float(img1.PixelSpacing[1]), float(sliceThickness)]
    dcmArray = np.zeros(imgSize, dtype=img1.pixel_array.dtype)

    # Load all DICOM slices and pool them into one variable (dcmArray)
    for i in range(len(listDcmFiles)):
        ds = read_file(listDcmFiles[i])
        print(ds.SeriesInstanceUID)
        dcmArray[:, :, i] = ds.pixel_array  
    return dcmArray

def loadDicomSeriesITK(dirName):
    listDcmFiles = os.listdir(dirName)
    listDcmFiles = [i for i in listDcmFiles if i.endswith('.dcm')]
    listDcmFiles.sort()
    dcmFile = os.path.join(dirName, listDcmFiles[0])

    dcm1 = read_file(dcmFile)
    seriesID = dcm1.SeriesInstanceUID
    original_image = sitk.ReadImage(sitk.ImageSeriesReader_GetGDCMSeriesFileNames(dirName,seriesID))

    return original_image

def getNumpyArrayFromITK(dcm):
    # Note that we flip the first and third axes because ITK
    # returns the array in the following axis order: [z, x, y]
    arr = sitk.GetArrayFromImage(dcm)
    if arr.ndim == 2:
        arr = np.moveaxis(arr,[0,1],[0,1])
    elif arr.ndim ==3:
        arr = np.moveaxis(arr,[0,1,2],[2,0,1])
    else:
        raise ValueError("Number of dimensions of the image are neither 2 nor 3. Check")
    return arr

def _loadNumbersInRowsFromTextFile(idxFname, dtype=np.float32):
    assert os.path.exists(idxFname), "File doesn't exist, check path"

    with open(idxFname, "r") as f:
        idx = f.read().split('\n')
        idx.remove('')
    idxLen = len(idx[0].split())
    indices = np.zeros((len(idx), idxLen), dtype=dtype)
    for item in idx:
        indices[idx.index(item),:] = dtype(item.split())
    return indices

def loadIndexFile(idxFname):
    assert os.path.exists(idxFname), "File doesn't exist, check path"
    assert idxFname.endswith('.txt'), "Size file extension must be .txt, check"
    indices = np.loadtxt(idxFname).astype(np.uint16)
    if indices.ndim==1:
        indices = indices[:,np.newaxis]
        indices = indices.T
    return indices[:,[1,0,2]]

def loadPhysicalPointsFile(ptsFname):
    assert os.path.exists(ptsFname), "File doesn't exist, check path"
    assert ptsFname.endswith('.txt'), "Size file extension must be .txt, check"
    return _loadNumbersInRowsFromTextFile(ptsFname, dtype=np.float32)

def loadSizesFile(szFname):
    assert os.path.exists(szFname), "File doesn't exist, check path"
    assert szFname.endswith('.txt'), "Size file extension must be .txt, check"
    size = np.loadtxt(szFname, ndmin=2)
    if size.shape[1]==2:
        size = (size[:,0]+size[:,1])*0.5
    return size

def loadMaskImageNifti(mskFname):
    assert os.path.exists(mskFname), "File doesn't exist, check path"
    assert mskFname.endswith('.nii.gz'), "Mask file extension must be .nii.gz, check"
    return sitk.ReadImage(mskFname)

def plotImageAndMaskSlice(img, mask, sliceNum):
    sliceNum  = np.int(sliceNum)
    img       = sitk.GetArrayFromImage( img[:,:,sliceNum]).astype(np.float)
    msk       = sitk.GetArrayFromImage(mask[:,:,sliceNum]).astype(np.float)
    maskAlpha = colorMaskWithAlpha(msk)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.imshow(maskAlpha)
    ax.axis('off')
    plt.show()

def plotImageAndMaskSliceForIdx(dcm, mask, Idx):
    Idx       = [int(i) for i in Idx]
    sliceNum  = Idx[2]
    Idx       = Idx[:-1] # Drop sliceNum

    img       = getNumpyArrayFromITK( dcm[:,:,sliceNum]).astype(np.float)
    msk       = getNumpyArrayFromITK(mask[:,:,sliceNum]).astype(np.uint8)

    if msk[Idx[0], Idx[1]] == 0:
        msk[:,:] = 0
    else:
        msk = msk==msk[Idx[0], Idx[1]]
    msk  = msk.astype(np.float)
    msk  = colorMaskWithAlpha(msk)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.imshow(msk)
    ax.axis('off')
    plt.show()

def convertImageToUINT8(image):
    image = image.astype(np.float)
    image = image-np.min(image)
    image = image/np.max(image)
    image = image*255.01
    return image.astype(np.uint8)

def convertGrayTo3Channel(gray):
    assert isinstance(gray, np.ndarray), "Image must be numpy array"
    assert gray.ndim==2, "Image must be 2D"
    m,n = gray.shape
    out = np.empty((m,n,3), dtype=gray.dtype)
    gray = gray[:,:,np.newaxis]
    out[:,:,:3] = gray
    return out

def getBoundaryOfBinaryBlob(img):
    assert isinstance(img, np.ndarray), "Input image must be a numpy array"
    assert img.ndim==2, "Image must be 2D"
    assert img.dtype==np.uint8, "Image data type must be np.uint8 with 0-255 scaling"

    imgEroded = np.copy(img) 
    kernel = np.ones((3,3), dtype=np.uint8)
    cv.erode(img, kernel, imgEroded, iterations=1)
    return img-imgEroded

def getIntersectionBoundaryAndLines(boundary, lines):
    assert isinstance(boundary, np.ndarray), "Boundary must be np.ndarray"
    assert isinstance(lines,    np.ndarray),    "Lines must be np.ndarray"
    assert boundary.dtype==np.bool, "Boundary dtype must be np.bool"
    assert lines.dtype==np.bool, "Lines dtype must be np.bool"

    x,y = _removeAmbiguityFromIntersecBoundaryLines(boundary & lines)
    if len(x)<4:
        raise ValueError("_removeAmbiguityFromIntersecBoundaryLines returned < 4 points. Check")
    return x, y

def _removeAmbiguityFromIntersecBoundaryLines(intersec):
    '''This function removes all but 1 point from connected
    components of the intersection image. This ensures there is
    only 1 point for intersection'''
    x, y = np.nonzero(intersec)
    numPts = len(x)
    if numPts==4:
        return x,y
    elif numPts>4:
        # Calculate distance matrix (L1 distance)
        distMat = np.zeros(shape=(numPts, numPts), dtype=np.float)
        for i in range(numPts):
            for j in range(i+1, numPts):
                distMat[i,j] = np.abs(x[i]-x[j]) + np.abs(y[i]-y[j])
        distMat = distMat + distMat.T
        # Identify dups to be removed - We do this by keeping a track of the
        # "referrer" and a "duplicate", through a list of lists (refdup)
        # If a refererrer asks a duplicate to be tagged as such (ref, dup) pair
        # We check if (dup, ref) pair already exists using refdup.

        dups = []
        refdup = [[] for i in range(numPts)]
        for i in range(numPts):
            for j in range(numPts):
                eps = 0.000001
                if 1-eps< distMat[i,j] < 2+eps:
                    # If duplicate not in dups and (dup, ref) not in refdup 
                    if j not in dups and i not in refdup[j]:
                        # Add (ref,dup) pair to refdup
                        refdup[i] = refdup[i]+[j]

                        # Add dup to the list of duplicates to be removed
                        dups = dups + [j]
        
        # Remove duplicates and return
        if dups is not []:
            x = np.delete(x, dups)
            y = np.delete(y, dups)
        return x, y
    else:
        raise ValueError("Less than 4 intersection points between boundary and lines. Check")
#
class lymphInfo:
    '''A class to hold info on lymph nodes found in a subject's scan
    Attributes:
        id: Subject ID, string
        maskFname: Mask file full path
        szFname: Size file full path
        indFname: indices file full path
        count: Number of lymph nodes, integer
        size: List of (list of floats storing tumor size)
            Tumor size is measured as average of short and long RECIST diameters in mm.
        mask: 3D numpy array storing mask pixel value
        indices: 3D numpy array storing x,y,z indices of 
        pixelVals: 1D Array, same length as (size), stores the mask value at the indices
        sizeListPerLN: A list of 2D numpy arrays, indexed by the lymph node count. 
            Each 2d arr stores z-index of slice where a lymph node was found, 
            and its extent in number of pixels.
        lnCountAboveThresh: 1D array with uint16 #elems=count.
            stores the number of slices that exceed a '#pixels-threshold'
            corresponding to each lymp node. #pixels-threshold is obtained
            as a fraction (0.8 default) of the pixels in the slice given by indices.
        lnCountThresh: 1D array of floats with #elems=count.
            Stores the threshold used to calculate lnCountAboveThresh.
    '''
    # We need access to mask, sizes, and indices files.
    def __init__(self, id='', maskFname='', szFname='', indFname=''):
        # Ensure inputs are valid before assignment
        assert isinstance(id, str), 'id must be of type str'
        assert isinstance(maskFname, str), 'maskFname must be of type str'
        assert isinstance(szFname,   str), 'szFname must be of type str'
        assert isinstance(indFname,  str), 'indFname must be of type str'

        assert maskFname.endswith('.nii.gz'), 'mask file name extension must be .nii.gz'
        assert szFname.endswith('.txt'), 'Size file name must end with .txt'
        assert indFname.endswith('.txt'), 'Indices file name must end with .txt'

        assert os.path.exists(maskFname), "Mask file doesn't exist, check path"
        assert os.path.exists(szFname), "Size file doesn't exist, check path"
        assert os.path.exists(indFname), "Indices file doesn't exist, check path"

        self.id = id
        self.maskFname = maskFname
        self.szFname   = szFname
        self.indFname  = indFname

        self.loadData()
        self.getSizeListPerLN()
        self.countLNinstanceAboveThresh(thresh=0.8)
    
    def loadData(self):
        '''The function loads the size, indices, and mask and calculates pixelVals and count'''
        self.size    = loadSizesFile(self.szFname)
        self.indices = loadIndexFile(self.indFname)
        self.mask    = loadMaskImageNifti(self.maskFname)
        self.mask    = getNumpyArrayFromITK(self.mask).astype(np.uint8)
    
        self.count = len(self.size)
        self.pixelVals = np.zeros(shape=(self.count, ), dtype=np.uint8)
        for i in range(self.count):
            # Here we load the pixel value at indices locations
            # on the mask.
            x,y,z = self.indices[i,:]
            self.pixelVals[i] = self.mask[x,y,z]

    def getSizeListPerLN(self):
        # For each lymph node, find out the slices that have non-zero
        # number of voxels across the volume. Store the number of voxels
        # and the corresponding slice index.
        self.sizeListPerLN = [np.zeros(shape=(1,2), dtype=np.float)]*self.count

        # For each slice in mask
        for sliceNum in range(self.mask.shape[2]):
            sliceXY = self.mask[:,:,sliceNum] # Get slice
            for i in range(self.count): # For all lymph nodes
                pixVal      = self.pixelVals[i] # Lymph node pixel value
                pixValCount = np.count_nonzero(sliceXY==pixVal) # Is this ln pixval present in current slice?
                if pixValCount: 
                    # If the lymph node is present in this slice, 
                    # store the slice number and the number of pixels
                    self.sizeListPerLN[i] = np.append(self.sizeListPerLN[i], np.array([[sliceNum, pixValCount]]), axis=0)

        for i in range(self.count):
            self.sizeListPerLN[i] = self.sizeListPerLN[i][1:]
        
    def countLNinstanceAboveThresh(self, thresh=0.8):
        zInd = self.indices[:,2] # Load the z indices
        self.lnCountAboveThresh = np.zeros(shape=(self.count,), dtype=np.uint16)
        self.lnCountThresh      = np.zeros(shape=(self.count,), dtype=np.float)

        for i in range(self.count):
            # Get the slice number and pixel count list for each ln
            instanceList = self.sizeListPerLN[i] 
            # look up where the slice number occurs in the instance list
            ind    = instanceList[:,0].tolist().index( zInd[i] )
            # Calculate a cutoff - slices with pixel count exceeding this are counted
            cutoff = thresh * instanceList[ind,1]
            aboveThresh = np.count_nonzero( instanceList[:,1]>=cutoff )
            self.lnCountAboveThresh[i] = aboveThresh
            self.lnCountThresh[i] = cutoff
#