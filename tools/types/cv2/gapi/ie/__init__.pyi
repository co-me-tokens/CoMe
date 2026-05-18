from __future__ import annotations
from . import detail
__all__: list[str] = ['ASYNC', 'Async', 'PyParams', 'SYNC', 'Sync', 'TRAIT_AS_IMAGE', 'TRAIT_AS_TENSOR', 'TraitAs_IMAGE', 'TraitAs_TENSOR', 'detail', 'params']
class PyParams:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def cfgBatchSize(size) -> retval:
        """
        .
        """
    @staticmethod
    def cfgNumRequests(nireq) -> retval:
        """
        .
        """
    @staticmethod
    def constInput(*args, **kwargs) -> retval:
        """
        .
        """
    def __repr__(self):
        """
        Return repr(self).
        """
def params(tag, model, weights, device) -> retval:
    """
    .   
    
    
    
    params(tag, model, device) -> retval
    .
    """
ASYNC: int = 1
Async: int = 1
SYNC: int = 0
Sync: int = 0
TRAIT_AS_IMAGE: int = 1
TRAIT_AS_TENSOR: int = 0
TraitAs_IMAGE: int = 1
TraitAs_TENSOR: int = 0
