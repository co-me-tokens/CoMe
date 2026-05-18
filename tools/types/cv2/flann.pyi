from __future__ import annotations
__all__: list[str] = ['FLANN_INDEX_TYPE_16S', 'FLANN_INDEX_TYPE_16U', 'FLANN_INDEX_TYPE_32F', 'FLANN_INDEX_TYPE_32S', 'FLANN_INDEX_TYPE_64F', 'FLANN_INDEX_TYPE_8S', 'FLANN_INDEX_TYPE_8U', 'FLANN_INDEX_TYPE_ALGORITHM', 'FLANN_INDEX_TYPE_BOOL', 'FLANN_INDEX_TYPE_STRING', 'Index', 'LAST_VALUE_FLANN_INDEX_TYPE']
class Index:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def build(*args, **kwargs) -> None:
        """
        .
        """
    @staticmethod
    def getAlgorithm() -> retval:
        """
        .
        """
    @staticmethod
    def getDistance() -> retval:
        """
        .
        """
    @staticmethod
    def knnSearch(*args, **kwargs) -> indices, dists:
        """
        .
        """
    @staticmethod
    def load(features, filename) -> retval:
        """
        .
        """
    @staticmethod
    def radiusSearch(*args, **kwargs) -> retval, indices, dists:
        """
        .
        """
    @staticmethod
    def release() -> None:
        """
        .
        """
    @staticmethod
    def save(filename) -> None:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
FLANN_INDEX_TYPE_16S: int = 3
FLANN_INDEX_TYPE_16U: int = 2
FLANN_INDEX_TYPE_32F: int = 5
FLANN_INDEX_TYPE_32S: int = 4
FLANN_INDEX_TYPE_64F: int = 6
FLANN_INDEX_TYPE_8S: int = 1
FLANN_INDEX_TYPE_8U: int = 0
FLANN_INDEX_TYPE_ALGORITHM: int = 9
FLANN_INDEX_TYPE_BOOL: int = 8
FLANN_INDEX_TYPE_STRING: int = 7
LAST_VALUE_FLANN_INDEX_TYPE: int = 9
