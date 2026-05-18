from __future__ import annotations
from collections import namedtuple
import cv2 as cv2
from cv2 import utils as _native
import typing
from . import fs
from . import nested
__all__: list[str] = ['ClassWithKeywordProperties', 'NativeMethodPatchedResult', 'cv2', 'dumpBool', 'dumpCString', 'dumpDouble', 'dumpFloat', 'dumpInputArray', 'dumpInputArrayOfArrays', 'dumpInputOutputArray', 'dumpInputOutputArrayOfArrays', 'dumpInt', 'dumpInt64', 'dumpRange', 'dumpRect', 'dumpRotatedRect', 'dumpSizeT', 'dumpString', 'dumpTermCriteria', 'dumpVec2i', 'dumpVectorOfDouble', 'dumpVectorOfInt', 'dumpVectorOfRect', 'fs', 'generateVectorOfInt', 'generateVectorOfMat', 'generateVectorOfRect', 'namedtuple', 'nested', 'testAsyncArray', 'testAsyncException', 'testOverloadResolution', 'testOverwriteNativeMethod', 'testRaiseGeneralException', 'testReservedKeywordConversion', 'testRotatedRect', 'testRotatedRectVector']
class ClassWithKeywordProperties:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
class NativeMethodPatchedResult(tuple):
    """
    NativeMethodPatchedResult(py, native)
    """
    __match_args__: typing.ClassVar[tuple] = ('py', 'native')
    __slots__: typing.ClassVar[tuple] = tuple()
    _field_defaults: typing.ClassVar[dict] = {}
    _fields: typing.ClassVar[tuple] = ('py', 'native')
    @staticmethod
    def __new__(_cls, py, native):
        """
        Create new instance of NativeMethodPatchedResult(py, native)
        """
    @classmethod
    def _make(cls, iterable):
        """
        Make a new NativeMethodPatchedResult object from a sequence or iterable
        """
    def __getnewargs__(self):
        """
        Return self as a plain tuple.  Used by copy and pickle.
        """
    def __repr__(self):
        """
        Return a nicely formatted representation string
        """
    def _asdict(self):
        """
        Return a new dict which maps field names to their values.
        """
    def _replace(self, **kwds):
        """
        Return a new NativeMethodPatchedResult object replacing specified fields with new values
        """
def dumpBool(argument) -> retval:
    """
    .
    """
def dumpCString(argument) -> retval:
    """
    .
    """
def dumpDouble(argument) -> retval:
    """
    .
    """
def dumpFloat(argument) -> retval:
    """
    .
    """
def dumpInputArray(argument) -> retval:
    """
    .
    """
def dumpInputArrayOfArrays(argument) -> retval:
    """
    .
    """
def dumpInputOutputArray(argument) -> retval, argument:
    """
    .
    """
def dumpInputOutputArrayOfArrays(argument) -> retval, argument:
    """
    .
    """
def dumpInt(argument) -> retval:
    """
    .
    """
def dumpInt64(argument) -> retval:
    """
    .
    """
def dumpRange(argument) -> retval:
    """
    .
    """
def dumpRect(argument) -> retval:
    """
    .
    """
def dumpRotatedRect(argument) -> retval:
    """
    .
    """
def dumpSizeT(argument) -> retval:
    """
    .
    """
def dumpString(argument) -> retval:
    """
    .
    """
def dumpTermCriteria(argument) -> retval:
    """
    .
    """
def dumpVec2i(*args, **kwargs) -> retval:
    """
    .
    """
def dumpVectorOfDouble(vec) -> retval:
    """
    .
    """
def dumpVectorOfInt(vec) -> retval:
    """
    .
    """
def dumpVectorOfRect(vec) -> retval:
    """
    .
    """
def generateVectorOfInt(len) -> vec:
    """
    .
    """
def generateVectorOfMat(*args, **kwargs) -> vec:
    """
    .
    """
def generateVectorOfRect(len) -> vec:
    """
    .
    """
def testAsyncArray(argument) -> retval:
    """
    .
    """
def testAsyncException() -> retval:
    """
    .
    """
def testOverloadResolution(*args, **kwargs) -> retval:
    """
    .   
    
    
    
    testOverloadResolution(rect) -> retval
    .
    """
def testOverwriteNativeMethod(arg):
    ...
def testRaiseGeneralException() -> None:
    """
    .
    """
def testReservedKeywordConversion(*args, **kwargs) -> retval:
    """
    .
    """
def testRotatedRect(x, y, w, h, angle) -> retval:
    """
    .
    """
def testRotatedRectVector(x, y, w, h, angle) -> retval:
    """
    .
    """
