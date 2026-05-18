from __future__ import annotations
__all__: list[str] = ['CALIB_CHECK_COND', 'CALIB_FIX_FOCAL_LENGTH', 'CALIB_FIX_INTRINSIC', 'CALIB_FIX_K1', 'CALIB_FIX_K2', 'CALIB_FIX_K3', 'CALIB_FIX_K4', 'CALIB_FIX_PRINCIPAL_POINT', 'CALIB_FIX_SKEW', 'CALIB_RECOMPUTE_EXTRINSIC', 'CALIB_USE_INTRINSIC_GUESS', 'CALIB_ZERO_DISPARITY', 'calibrate', 'distortPoints', 'estimateNewCameraMatrixForUndistortRectify', 'initUndistortRectifyMap', 'projectPoints', 'stereoCalibrate', 'stereoRectify', 'undistortImage', 'undistortPoints']
def calibrate(*args, **kwargs) -> retval, K, D, rvecs, tvecs:
    """
    .   @brief Performs camera calibration
    .   
    .       @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
    .       coordinate space.
    .       @param imagePoints vector of vectors of the projections of calibration pattern points.
    .       imagePoints.size() and objectPoints.size() and imagePoints[i].size() must be equal to
    .       objectPoints[i].size() for each i.
    .       @param image_size Size of the image used only to initialize the camera intrinsic matrix.
    .       @param K Output 3x3 floating-point camera intrinsic matrix
    .       \\f$\\cameramatrix{A}\\f$ . If
    .       @ref fisheye::CALIB_USE_INTRINSIC_GUESS is specified, some or all of fx, fy, cx, cy must be
    .       initialized before calling the function.
    .       @param D Output vector of distortion coefficients \\f$\\distcoeffsfisheye\\f$.
    .       @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
    .       That is, each k-th rotation vector together with the corresponding k-th translation vector (see
    .       the next output parameter description) brings the calibration pattern from the model coordinate
    .       space (in which object points are specified) to the world coordinate space, that is, a real
    .       position of the calibration pattern in the k-th pattern view (k=0.. *M* -1).
    .       @param tvecs Output vector of translation vectors estimated for each pattern view.
    .       @param flags Different flags that may be zero or a combination of the following values:
    .       -    @ref fisheye::CALIB_USE_INTRINSIC_GUESS  cameraMatrix contains valid initial values of
    .       fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    .       center ( imageSize is used), and focal distances are computed in a least-squares fashion.
    .       -    @ref fisheye::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
    .       of intrinsic optimization.
    .       -    @ref fisheye::CALIB_CHECK_COND  The functions will check validity of condition number.
    .       -    @ref fisheye::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
    .       -    @ref fisheye::CALIB_FIX_K1,..., @ref fisheye::CALIB_FIX_K4 Selected distortion coefficients
    .       are set to zeros and stay zero.
    .       -    @ref fisheye::CALIB_FIX_PRINCIPAL_POINT  The principal point is not changed during the global
    .   optimization. It stays at the center or at a different location specified when @ref fisheye::CALIB_USE_INTRINSIC_GUESS is set too.
    .       -    @ref fisheye::CALIB_FIX_FOCAL_LENGTH The focal length is not changed during the global
    .   optimization. It is the \\f$max(width,height)/\\pi\\f$ or the provided \\f$f_x\\f$, \\f$f_y\\f$ when @ref fisheye::CALIB_USE_INTRINSIC_GUESS is set too.
    .       @param criteria Termination criteria for the iterative optimization algorithm.
    """
def distortPoints(*args, **kwargs) -> distorted:
    """
    .   @brief Distorts 2D points using fisheye model.
    .   
    .       @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector\\<Point2f\\> ), where N is
    .       the number of points in the view.
    .       @param K Camera intrinsic matrix \\f$cameramatrix{K}\\f$.
    .       @param D Input vector of distortion coefficients \\f$\\distcoeffsfisheye\\f$.
    .       @param alpha The skew coefficient.
    .       @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\\<Point2f\\> .
    .   
    .       Note that the function assumes the camera intrinsic matrix of the undistorted points to be identity.
    .       This means if you want to distort image points you have to multiply them with \\f$K^{-1}\\f$.
    """
def estimateNewCameraMatrixForUndistortRectify(*args, **kwargs) -> P:
    """
    .   @brief Estimates new camera intrinsic matrix for undistortion or rectification.
    .   
    .       @param K Camera intrinsic matrix \\f$cameramatrix{K}\\f$.
    .       @param image_size Size of the image
    .       @param D Input vector of distortion coefficients \\f$\\distcoeffsfisheye\\f$.
    .       @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    .       1-channel or 1x1 3-channel
    .       @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    .       @param balance Sets the new focal length in range between the min focal length and the max focal
    .       length. Balance is in range of [0, 1].
    .       @param new_size the new size
    .       @param fov_scale Divisor for new focal length.
    """
def initUndistortRectifyMap(*args, **kwargs) -> map1, map2:
    """
    .   @brief Computes undistortion and rectification maps for image transform by #remap. If D is empty zero
    .       distortion is used, if R or P is empty identity matrixes are used.
    .   
    .       @param K Camera intrinsic matrix \\f$cameramatrix{K}\\f$.
    .       @param D Input vector of distortion coefficients \\f$\\distcoeffsfisheye\\f$.
    .       @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    .       1-channel or 1x1 3-channel
    .       @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    .       @param size Undistorted image size.
    .       @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See #convertMaps
    .       for details.
    .       @param map1 The first output map.
    .       @param map2 The second output map.
    """
def projectPoints(*args, **kwargs) -> imagePoints, jacobian:
    """
    .   @overload
    """
def stereoCalibrate(*args, **kwargs) -> retval, K1, D1, K2, D2, R, T, rvecs, tvecs:
    """
    .   @brief Performs stereo calibration
    .   
    .       @param objectPoints Vector of vectors of the calibration pattern points.
    .       @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
    .       observed by the first camera.
    .       @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
    .       observed by the second camera.
    .       @param K1 Input/output first camera intrinsic matrix:
    .       \\f$\\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\\f$ , \\f$j = 0,\\, 1\\f$ . If
    .       any of @ref fisheye::CALIB_USE_INTRINSIC_GUESS , @ref fisheye::CALIB_FIX_INTRINSIC are specified,
    .       some or all of the matrix components must be initialized.
    .       @param D1 Input/output vector of distortion coefficients \\f$\\distcoeffsfisheye\\f$ of 4 elements.
    .       @param K2 Input/output second camera intrinsic matrix. The parameter is similar to K1 .
    .       @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
    .       similar to D1 .
    .       @param imageSize Size of the image used only to initialize camera intrinsic matrix.
    .       @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
    .       @param T Output translation vector between the coordinate systems of the cameras.
    .       @param rvecs Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in the
    .       coordinate system of the first camera of the stereo pair (e.g. std::vector<cv::Mat>). More in detail, each
    .       i-th rotation vector together with the corresponding i-th translation vector (see the next output parameter
    .       description) brings the calibration pattern from the object coordinate space (in which object points are
    .       specified) to the camera coordinate space of the first camera of the stereo pair. In more technical terms,
    .       the tuple of the i-th rotation and translation vector performs a change of basis from object coordinate space
    .       to camera coordinate space of the first camera of the stereo pair.
    .       @param tvecs Output vector of translation vectors estimated for each pattern view, see parameter description
    .       of previous output parameter ( rvecs ).
    .       @param flags Different flags that may be zero or a combination of the following values:
    .       -    @ref fisheye::CALIB_FIX_INTRINSIC  Fix K1, K2? and D1, D2? so that only R, T matrices
    .       are estimated.
    .       -    @ref fisheye::CALIB_USE_INTRINSIC_GUESS  K1, K2 contains valid initial values of
    .       fx, fy, cx, cy that are optimized further. Otherwise, (cx, cy) is initially set to the image
    .       center (imageSize is used), and focal distances are computed in a least-squares fashion.
    .       -    @ref fisheye::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
    .       of intrinsic optimization.
    .       -    @ref fisheye::CALIB_CHECK_COND  The functions will check validity of condition number.
    .       -    @ref fisheye::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
    .       -   @ref fisheye::CALIB_FIX_K1,..., @ref fisheye::CALIB_FIX_K4 Selected distortion coefficients are set to zeros and stay
    .       zero.
    .       @param criteria Termination criteria for the iterative optimization algorithm.
    
    
    
    stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize[, R[, T[, flags[, criteria]]]]) -> retval, K1, D1, K2, D2, R, T
    .
    """
def stereoRectify(*args, **kwargs) -> R1, R2, P1, P2, Q:
    """
    .   @brief Stereo rectification for fisheye camera model
    .   
    .       @param K1 First camera intrinsic matrix.
    .       @param D1 First camera distortion parameters.
    .       @param K2 Second camera intrinsic matrix.
    .       @param D2 Second camera distortion parameters.
    .       @param imageSize Size of the image used for stereo calibration.
    .       @param R Rotation matrix between the coordinate systems of the first and the second
    .       cameras.
    .       @param tvec Translation vector between coordinate systems of the cameras.
    .       @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
    .       @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
    .       @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
    .       camera.
    .       @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
    .       camera.
    .       @param Q Output \\f$4 \\times 4\\f$ disparity-to-depth mapping matrix (see #reprojectImageTo3D ).
    .       @param flags Operation flags that may be zero or @ref fisheye::CALIB_ZERO_DISPARITY . If the flag is set,
    .       the function makes the principal points of each camera have the same pixel coordinates in the
    .       rectified views. And if the flag is not set, the function may still shift the images in the
    .       horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the
    .       useful image area.
    .       @param newImageSize New image resolution after rectification. The same size should be passed to
    .       #initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
    .       is passed (default), it is set to the original imageSize . Setting it to larger value can help you
    .       preserve details in the original image, especially when there is a big radial distortion.
    .       @param balance Sets the new focal length in range between the min focal length and the max focal
    .       length. Balance is in range of [0, 1].
    .       @param fov_scale Divisor for new focal length.
    """
def undistortImage(*args, **kwargs) -> undistorted:
    """
    .   @brief Transforms an image to compensate for fisheye lens distortion.
    .   
    .       @param distorted image with fisheye lens distortion.
    .       @param undistorted Output image with compensated fisheye lens distortion.
    .       @param K Camera intrinsic matrix \\f$cameramatrix{K}\\f$.
    .       @param D Input vector of distortion coefficients \\f$\\distcoeffsfisheye\\f$.
    .       @param Knew Camera intrinsic matrix of the distorted image. By default, it is the identity matrix but you
    .       may additionally scale and shift the result by using a different matrix.
    .       @param new_size the new size
    .   
    .       The function transforms an image to compensate radial and tangential lens distortion.
    .   
    .       The function is simply a combination of #fisheye::initUndistortRectifyMap (with unity R ) and #remap
    .       (with bilinear interpolation). See the former function for details of the transformation being
    .       performed.
    .   
    .       See below the results of undistortImage.
    .          -   a\\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
    .               k_4, k_5, k_6) of distortion were optimized under calibration)
    .           -   b\\) result of #fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
    .               k_3, k_4) of fisheye distortion were optimized under calibration)
    .           -   c\\) original image was captured with fisheye lens
    .   
    .       Pictures a) and b) almost the same. But if we consider points of image located far from the center
    .       of image, we can notice that on image a) these points are distorted.
    .   
    .       ![image](pics/fisheye_undistorted.jpg)
    """
def undistortPoints(*args, **kwargs) -> undistorted:
    """
    .   @brief Undistorts 2D points using fisheye model
    .   
    .       @param distorted Array of object points, 1xN/Nx1 2-channel (or vector\\<Point2f\\> ), where N is the
    .       number of points in the view.
    .       @param K Camera intrinsic matrix \\f$cameramatrix{K}\\f$.
    .       @param D Input vector of distortion coefficients \\f$\\distcoeffsfisheye\\f$.
    .       @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    .       1-channel or 1x1 3-channel
    .       @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    .       @param criteria Termination criteria
    .       @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector\\<Point2f\\> .
    """
CALIB_CHECK_COND: int = 4
CALIB_FIX_FOCAL_LENGTH: int = 2048
CALIB_FIX_INTRINSIC: int = 256
CALIB_FIX_K1: int = 16
CALIB_FIX_K2: int = 32
CALIB_FIX_K3: int = 64
CALIB_FIX_K4: int = 128
CALIB_FIX_PRINCIPAL_POINT: int = 512
CALIB_FIX_SKEW: int = 8
CALIB_RECOMPUTE_EXTRINSIC: int = 2
CALIB_USE_INTRINSIC_GUESS: int = 1
CALIB_ZERO_DISPARITY: int = 1024
