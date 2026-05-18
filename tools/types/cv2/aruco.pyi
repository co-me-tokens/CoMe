from __future__ import annotations
import cv2
__all__: list[str] = ['ArucoDetector', 'Board', 'CORNER_REFINE_APRILTAG', 'CORNER_REFINE_CONTOUR', 'CORNER_REFINE_NONE', 'CORNER_REFINE_SUBPIX', 'CharucoBoard', 'CharucoDetector', 'CharucoParameters', 'DICT_4X4_100', 'DICT_4X4_1000', 'DICT_4X4_250', 'DICT_4X4_50', 'DICT_5X5_100', 'DICT_5X5_1000', 'DICT_5X5_250', 'DICT_5X5_50', 'DICT_6X6_100', 'DICT_6X6_1000', 'DICT_6X6_250', 'DICT_6X6_50', 'DICT_7X7_100', 'DICT_7X7_1000', 'DICT_7X7_250', 'DICT_7X7_50', 'DICT_APRILTAG_16H5', 'DICT_APRILTAG_16h5', 'DICT_APRILTAG_25H9', 'DICT_APRILTAG_25h9', 'DICT_APRILTAG_36H10', 'DICT_APRILTAG_36H11', 'DICT_APRILTAG_36h10', 'DICT_APRILTAG_36h11', 'DICT_ARUCO_ORIGINAL', 'DetectorParameters', 'Dictionary', 'Dictionary_getBitsFromByteList', 'Dictionary_getByteListFromBits', 'GridBoard', 'RefineParameters', 'drawDetectedCornersCharuco', 'drawDetectedDiamonds', 'drawDetectedMarkers', 'extendDictionary', 'generateImageMarker', 'getPredefinedDictionary']
class ArucoDetector(cv2.Algorithm):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def detectMarkers(*args, **kwargs) -> corners, ids, rejectedImgPoints:
        """
        .   @brief Basic marker detection
        .        *
        .        * @param image input image
        .        * @param corners vector of detected marker corners. For each marker, its four corners
        .        * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
        .        * the dimensions of this array is Nx4. The order of the corners is clockwise.
        .        * @param ids vector of identifiers of the detected markers. The identifier is of type int
        .        * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
        .        * The identifiers have the same order than the markers in the imgPoints array.
        .        * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
        .        * correct codification. Useful for debugging purposes.
        .        *
        .        * Performs marker detection in the input image. Only markers included in the specific dictionary
        .        * are searched. For each detected marker, it returns the 2D position of its corner in the image
        .        * and its corresponding identifier.
        .        * Note that this function does not perform pose estimation.
        .        * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
        .        * input image with corresponging camera model, if camera parameters are known
        .        * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
        """
    @staticmethod
    def getDetectorParameters() -> retval:
        """
        .
        """
    @staticmethod
    def getDictionary() -> retval:
        """
        .
        """
    @staticmethod
    def getRefineParameters() -> retval:
        """
        .
        """
    @staticmethod
    def read(fn) -> None:
        """
        .   @brief Reads algorithm parameters from a file storage
        """
    @staticmethod
    def refineDetectedMarkers(*args, **kwargs) -> detectedCorners, detectedIds, rejectedCorners, recoveredIdxs:
        """
        .   @brief Refind not detected markers based on the already detected and the board layout
        .        *
        .        * @param image input image
        .        * @param board layout of markers in the board.
        .        * @param detectedCorners vector of already detected marker corners.
        .        * @param detectedIds vector of already detected marker identifiers.
        .        * @param rejectedCorners vector of rejected candidates during the marker detection process.
        .        * @param cameraMatrix optional input 3x3 floating-point camera matrix
        .        * \\f$A = \\vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\\f$
        .        * @param distCoeffs optional vector of distortion coefficients
        .        * \\f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\\f$ of 4, 5, 8 or 12 elements
        .        * @param recoveredIdxs Optional array to returns the indexes of the recovered candidates in the
        .        * original rejectedCorners array.
        .        *
        .        * This function tries to find markers that were not detected in the basic detecMarkers function.
        .        * First, based on the current detected marker and the board layout, the function interpolates
        .        * the position of the missing markers. Then it tries to find correspondence between the reprojected
        .        * markers and the rejected candidates based on the minRepDistance and errorCorrectionRate parameters.
        .        * If camera parameters and distortion coefficients are provided, missing markers are reprojected
        .        * using projectPoint function. If not, missing marker projections are interpolated using global
        .        * homography, and all the marker corners in the board must have the same Z coordinate.
        """
    @staticmethod
    def setDetectorParameters(detectorParameters) -> None:
        """
        .
        """
    @staticmethod
    def setDictionary(dictionary) -> None:
        """
        .
        """
    @staticmethod
    def setRefineParameters(refineParameters) -> None:
        """
        .
        """
    @staticmethod
    def write(fs, name) -> None:
        """
        .   @brief simplified API for language bindings
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Board:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def generateImage(*args, **kwargs) -> img:
        """
        .   @brief Draw a planar board
        .        *
        .        * @param outSize size of the output image in pixels.
        .        * @param img output image with the board. The size of this image will be outSize
        .        * and the board will be on the center, keeping the board proportions.
        .        * @param marginSize minimum margins (in pixels) of the board in the output image
        .        * @param borderBits width of the marker borders.
        .        *
        .        * This function return the image of the board, ready to be printed.
        """
    @staticmethod
    def getDictionary() -> retval:
        """
        .   @brief return the Dictionary of markers employed for this board
        """
    @staticmethod
    def getIds() -> retval:
        """
        .   @brief vector of the identifiers of the markers in the board (should be the same size as objPoints)
        .        * @return vector of the identifiers of the markers
        """
    @staticmethod
    def getObjPoints() -> retval:
        """
        .   @brief return array of object points of all the marker corners in the board.
        .        *
        .        * Each marker include its 4 corners in this order:
        .        * -   objPoints[i][0] - left-top point of i-th marker
        .        * -   objPoints[i][1] - right-top point of i-th marker
        .        * -   objPoints[i][2] - right-bottom point of i-th marker
        .        * -   objPoints[i][3] - left-bottom point of i-th marker
        .        *
        .        * Markers are placed in a certain order - row by row, left to right in every row. For M markers, the size is Mx4.
        """
    @staticmethod
    def getRightBottomCorner() -> retval:
        """
        .   @brief get coordinate of the bottom right corner of the board, is set when calling the function create()
        """
    @staticmethod
    def matchImagePoints(*args, **kwargs) -> objPoints, imgPoints:
        """
        .   @brief Given a board configuration and a set of detected markers, returns the corresponding
        .        * image points and object points to call solvePnP()
        .        *
        .        * @param detectedCorners List of detected marker corners of the board.
        .        * For CharucoBoard class you can set list of charuco corners.
        .        * @param detectedIds List of identifiers for each marker or list of charuco identifiers for each corner.
        .        * For CharucoBoard class you can set list of charuco identifiers for each corner.
        .        * @param objPoints Vector of vectors of board marker points in the board coordinate space.
        .        * @param imgPoints Vector of vectors of the projections of board marker corner points.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class CharucoBoard(Board):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def checkCharucoCornersCollinear(charucoIds) -> retval:
        """
        .   @brief check whether the ChArUco markers are collinear
        .        *
        .        * @param charucoIds list of identifiers for each corner in charucoCorners per frame.
        .        * @return bool value, 1 (true) if detected corners form a line, 0 (false) if they do not.
        .        * solvePnP, calibration functions will fail if the corners are collinear (true).
        .        *
        .        * The number of ids in charucoIDs should be <= the number of chessboard corners in the board.
        .        * This functions checks whether the charuco corners are on a straight line (returns true, if so), or not (false).
        .        * Axis parallel, as well as diagonal and other straight lines detected.  Degenerate cases:
        .        * for number of charucoIDs <= 2,the function returns true.
        """
    @staticmethod
    def getChessboardCorners() -> retval:
        """
        .   @brief get CharucoBoard::chessboardCorners
        """
    @staticmethod
    def getChessboardSize() -> retval:
        """
        .
        """
    @staticmethod
    def getMarkerLength() -> retval:
        """
        .
        """
    @staticmethod
    def getSquareLength() -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class CharucoDetector(cv2.Algorithm):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def detectBoard(*args, **kwargs) -> charucoCorners, charucoIds, markerCorners, markerIds:
        """
        .   * @brief detect aruco markers and interpolate position of ChArUco board corners
        .        * @param image input image necesary for corner refinement. Note that markers are not detected and
        .        * should be sent in corners and ids parameters.
        .        * @param charucoCorners interpolated chessboard corners.
        .        * @param charucoIds interpolated chessboard corners identifiers.
        .        * @param markerCorners vector of already detected markers corners. For each marker, its four
        .        * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
        .        * dimensions of this array should be Nx4. The order of the corners should be clockwise.
        .        * If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        .        * @param markerIds list of identifiers for each marker in corners.
        .        *  If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        .        *
        .        * This function receives the detected markers and returns the 2D position of the chessboard corners
        .        * from a ChArUco board using the detected Aruco markers.
        .        *
        .        * If markerCorners and markerCorners are empty, the detectMarkers() will run and detect aruco markers and ids.
        .        *
        .        * If camera parameters are provided, the process is based in an approximated pose estimation, else it is based on local homography.
        .        * Only visible corners are returned. For each corner, its corresponding identifier is also returned in charucoIds.
        .        * @sa findChessboardCorners
        """
    @staticmethod
    def detectDiamonds(*args, **kwargs) -> diamondCorners, diamondIds, markerCorners, markerIds:
        """
        .   * @brief Detect ChArUco Diamond markers
        .        *
        .        * @param image input image necessary for corner subpixel.
        .        * @param diamondCorners output list of detected diamond corners (4 corners per diamond). The order
        .        * is the same than in marker corners: top left, top right, bottom right and bottom left. Similar
        .        * format than the corners returned by detectMarkers (e.g std::vector<std::vector<cv::Point2f> > ).
        .        * @param diamondIds ids of the diamonds in diamondCorners. The id of each diamond is in fact of
        .        * type Vec4i, so each diamond has 4 ids, which are the ids of the aruco markers composing the
        .        * diamond.
        .        * @param markerCorners list of detected marker corners from detectMarkers function.
        .        * If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        .        * @param markerIds list of marker ids in markerCorners.
        .        * If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        .        *
        .        * This function detects Diamond markers from the previous detected ArUco markers. The diamonds
        .        * are returned in the diamondCorners and diamondIds parameters. If camera calibration parameters
        .        * are provided, the diamond search is based on reprojection. If not, diamond search is based on
        .        * homography. Homography is faster than reprojection, but less accurate.
        """
    @staticmethod
    def getBoard() -> retval:
        """
        .
        """
    @staticmethod
    def getCharucoParameters() -> retval:
        """
        .
        """
    @staticmethod
    def getDetectorParameters() -> retval:
        """
        .
        """
    @staticmethod
    def getRefineParameters() -> retval:
        """
        .
        """
    @staticmethod
    def setBoard(board) -> None:
        """
        .
        """
    @staticmethod
    def setCharucoParameters(charucoParameters) -> None:
        """
        .
        """
    @staticmethod
    def setDetectorParameters(detectorParameters) -> None:
        """
        .
        """
    @staticmethod
    def setRefineParameters(refineParameters) -> None:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class CharucoParameters:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class DetectorParameters:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def readDetectorParameters(fn) -> retval:
        """
        .   @brief Read a new set of DetectorParameters from FileNode (use FileStorage.root()).
        """
    @staticmethod
    def writeDetectorParameters(*args, **kwargs) -> retval:
        """
        .   @brief Write a set of DetectorParameters to FileStorage
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class Dictionary:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def generateImageMarker(*args, **kwargs) -> _img:
        """
        .   @brief Generate a canonical marker image
        """
    @staticmethod
    def getBitsFromByteList(byteList, markerSize) -> retval:
        """
        .   @brief Transform list of bytes to matrix of bits
        """
    @staticmethod
    def getByteListFromBits(bits) -> retval:
        """
        .   @brief Transform matrix of bits to list of bytes in the 4 rotations
        """
    @staticmethod
    def getDistanceToId(*args, **kwargs) -> retval:
        """
        .   @brief Returns the distance of the input bits to the specific id.
        .        *
        .        * If allRotations is true, the four posible bits rotation are considered
        """
    @staticmethod
    def identify(onlyBits, maxCorrectionRate) -> retval, idx, rotation:
        """
        .   @brief Given a matrix of bits. Returns whether if marker is identified or not.
        .        *
        .        * It returns by reference the correct id (if any) and the correct rotation
        """
    @staticmethod
    def readDictionary(fn) -> retval:
        """
        .   @brief Read a new dictionary from FileNode.
        .        *
        .        * Dictionary format:\\n
        .        * nmarkers: 35\\n
        .        * markersize: 6\\n
        .        * maxCorrectionBits: 5\\n
        .        * marker_0: "101011111011111001001001101100000000"\\n
        .        * ...\\n
        .        * marker_34: "011111010000111011111110110101100101"
        """
    @staticmethod
    def writeDictionary(*args, **kwargs) -> None:
        """
        .   @brief Write a dictionary to FileStorage, format is the same as in readDictionary().
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class GridBoard(Board):
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def getGridSize() -> retval:
        """
        .
        """
    @staticmethod
    def getMarkerLength() -> retval:
        """
        .
        """
    @staticmethod
    def getMarkerSeparation() -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class RefineParameters:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def readRefineParameters(fn) -> retval:
        """
        .   @brief Read a new set of RefineParameters from FileNode (use FileStorage.root()).
        """
    @staticmethod
    def writeRefineParameters(*args, **kwargs) -> retval:
        """
        .   @brief Write a set of RefineParameters to FileStorage
        """
    def __repr__(self):
        """
        Return repr(self).
        """
def Dictionary_getBitsFromByteList(byteList, markerSize) -> retval:
    """
    .   @brief Transform list of bytes to matrix of bits
    """
def Dictionary_getByteListFromBits(bits) -> retval:
    """
    .   @brief Transform matrix of bits to list of bytes in the 4 rotations
    """
def drawDetectedCornersCharuco(*args, **kwargs) -> image:
    """
    .   * @brief Draws a set of Charuco corners
    .    * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
    .    * altered.
    .    * @param charucoCorners vector of detected charuco corners
    .    * @param charucoIds list of identifiers for each corner in charucoCorners
    .    * @param cornerColor color of the square surrounding each corner
    .    *
    .    * This function draws a set of detected Charuco corners. If identifiers vector is provided, it also
    .    * draws the id of each corner.
    """
def drawDetectedDiamonds(*args, **kwargs) -> image:
    """
    .   * @brief Draw a set of detected ChArUco Diamond markers
    .    *
    .    * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
    .    * altered.
    .    * @param diamondCorners positions of diamond corners in the same format returned by
    .    * detectCharucoDiamond(). (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
    .    * the dimensions of this array should be Nx4. The order of the corners should be clockwise.
    .    * @param diamondIds vector of identifiers for diamonds in diamondCorners, in the same format
    .    * returned by detectCharucoDiamond() (e.g. std::vector<Vec4i>).
    .    * Optional, if not provided, ids are not painted.
    .    * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
    .    * are calculated based on this one.
    .    *
    .    * Given an array of detected diamonds, this functions draws them in the image. The marker borders
    .    * are painted and the markers identifiers if provided.
    .    * Useful for debugging purposes.
    """
def drawDetectedMarkers(*args, **kwargs) -> image:
    """
    .   @brief Draw detected markers in image
    .    *
    .    * @param image input/output image. It must have 1 or 3 channels. The number of channels is not altered.
    .    * @param corners positions of marker corners on input image.
    .    * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
    .    * this array should be Nx4. The order of the corners should be clockwise.
    .    * @param ids vector of identifiers for markers in markersCorners .
    .    * Optional, if not provided, ids are not painted.
    .    * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
    .    * are calculated based on this one to improve visualization.
    .    *
    .    * Given an array of detected marker corners and its corresponding ids, this functions draws
    .    * the markers in the image. The marker borders are painted and the markers identifiers if provided.
    .    * Useful for debugging purposes.
    """
def extendDictionary(*args, **kwargs) -> retval:
    """
    .   @brief Extend base dictionary by new nMarkers
    .     *
    .     * @param nMarkers number of markers in the dictionary
    .     * @param markerSize number of bits per dimension of each markers
    .     * @param baseDictionary Include the markers in this dictionary at the beginning (optional)
    .     * @param randomSeed a user supplied seed for theRNG()
    .     *
    .     * This function creates a new dictionary composed by nMarkers markers and each markers composed
    .     * by markerSize x markerSize bits. If baseDictionary is provided, its markers are directly
    .     * included and the rest are generated based on them. If the size of baseDictionary is higher
    .     * than nMarkers, only the first nMarkers in baseDictionary are taken and no new marker is added.
    """
def generateImageMarker(*args, **kwargs) -> img:
    """
    .   @brief Generate a canonical marker image
    .    *
    .    * @param dictionary dictionary of markers indicating the type of markers
    .    * @param id identifier of the marker that will be returned. It has to be a valid id in the specified dictionary.
    .    * @param sidePixels size of the image in pixels
    .    * @param img output image with the marker
    .    * @param borderBits width of the marker border.
    .    *
    .    * This function returns a marker image in its canonical form (i.e. ready to be printed)
    """
def getPredefinedDictionary(dict) -> retval:
    """
    .   @brief Returns one of the predefined dictionaries referenced by DICT_*.
    """
CORNER_REFINE_APRILTAG: int = 3
CORNER_REFINE_CONTOUR: int = 2
CORNER_REFINE_NONE: int = 0
CORNER_REFINE_SUBPIX: int = 1
DICT_4X4_100: int = 1
DICT_4X4_1000: int = 3
DICT_4X4_250: int = 2
DICT_4X4_50: int = 0
DICT_5X5_100: int = 5
DICT_5X5_1000: int = 7
DICT_5X5_250: int = 6
DICT_5X5_50: int = 4
DICT_6X6_100: int = 9
DICT_6X6_1000: int = 11
DICT_6X6_250: int = 10
DICT_6X6_50: int = 8
DICT_7X7_100: int = 13
DICT_7X7_1000: int = 15
DICT_7X7_250: int = 14
DICT_7X7_50: int = 12
DICT_APRILTAG_16H5: int = 17
DICT_APRILTAG_16h5: int = 17
DICT_APRILTAG_25H9: int = 18
DICT_APRILTAG_25h9: int = 18
DICT_APRILTAG_36H10: int = 19
DICT_APRILTAG_36H11: int = 20
DICT_APRILTAG_36h10: int = 19
DICT_APRILTAG_36h11: int = 20
DICT_ARUCO_ORIGINAL: int = 16
