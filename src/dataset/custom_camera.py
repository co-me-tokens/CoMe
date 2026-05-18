"""Live camera dataset loader for multi-view geometry tasks.

Frames are captured by a dedicated worker process that owns the ``cv2.VideoCapture``
handle and keeps a bounded shared buffer of the most recent frames. Each iteration
waits until at least ``segment_length`` frames are available, uniformly samples a
segment from the current buffer snapshot, clears the buffer, and returns a
``MultiViewTask`` with ``scene_geometry=None``.

WARNING: This dataset starts its own multiprocessing worker and must only be used
from a single consumer with ``DataLoader(num_workers=0)`` or direct iteration.
Do not use it with DataLoader worker processes, random-access evaluation code, or
multi-process training pipelines.
"""

import math
import multiprocessing as mp
from multiprocessing.process import BaseProcess
import time
import typing as T

import cv2 as cv
import numpy as np
import torch

from ..interface.dataset import MultiViewTask
from ..interface.geometric_model import MultiViewInput


MultiViewStream = torch.utils.data.IterDataPipe[MultiViewTask]


# ==== Worker helpers ====

def _set_worker_error(error_state: T.Any, message: str) -> None:
    if error_state.get("message") is None:
        error_state["message"] = message


def _camera_worker(
    cam_id: int,
    freq_hz: float,
    buffer_size: int,
    shared_buffer: T.Any,
    buffer_lock: T.Any,
    error_state: T.Any,
    ready_event: T.Any,
    stop_event: T.Any,
) -> None:
    cap: cv.VideoCapture | None = None

    try:
        cap = cv.VideoCapture(cam_id)
        if not cap.isOpened():
            _set_worker_error(error_state, f"cv2.VideoCapture could not open camera id {cam_id}.")
            return

        ready_event.set()
        period_sec = 1.0 / freq_hz
        expected_shape: tuple[int, int, int] | None = None
        next_capture = time.monotonic()

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                _set_worker_error(error_state, f"Failed to read a frame from camera id {cam_id}.")
                return
            if frame.ndim != 3 or frame.shape[2] != 3:
                _set_worker_error(
                    error_state,
                    f"Camera id {cam_id} returned an invalid frame shape {tuple(frame.shape)}; expected HxWx3.",
                )
                return

            frame_shape = T.cast(tuple[int, int, int], tuple(frame.shape))
            if expected_shape is None:
                expected_shape = frame_shape
            elif frame_shape != expected_shape:
                _set_worker_error(
                    error_state,
                    f"Camera id {cam_id} returned inconsistent frame shape {frame_shape}; expected {expected_shape}.",
                )
                return

            with buffer_lock:
                if len(shared_buffer) >= buffer_size:
                    shared_buffer.pop(0)
                shared_buffer.append(np.ascontiguousarray(frame))

            next_capture += period_sec
            sleep_sec = next_capture - time.monotonic()
            if sleep_sec > 0.0 and stop_event.wait(timeout=sleep_sec):
                break
            if sleep_sec <= 0.0:
                next_capture = time.monotonic()
    except Exception as exc:
        _set_worker_error(error_state, f"Camera worker crashed for camera id {cam_id}: {exc!r}")
    finally:
        if cap is not None:
            cap.release()
        ready_event.set()


# ==== Dataset ====

class CameraStreamDataset(MultiViewStream):
    _READY_TIMEOUT_SEC = 5.0
    _JOIN_TIMEOUT_SEC = 2.0
    _POLL_INTERVAL_SEC = 0.01

    def __init__(
        self,
        cam_id: int = 0,
        segment_length: int = 1,
        freq_hz: float = 12.0,
        buffer_size: int = 128,
    ) -> None:
        super().__init__()

        if cam_id < 0:
            raise ValueError(f"cam_id must be >= 0, got {cam_id}")
        if segment_length < 1:
            raise ValueError(f"segment_length must be >= 1, got {segment_length}")
        if not math.isfinite(freq_hz) or freq_hz <= 0.0:
            raise ValueError(f"freq_hz must be a finite positive number, got {freq_hz}")
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        if buffer_size < segment_length:
            raise ValueError(
                f"buffer_size must be >= segment_length, got buffer_size={buffer_size}, segment_length={segment_length}"
            )

        self._cam_id = cam_id
        self._segment_length = segment_length
        self._freq_hz = freq_hz
        self._buffer_size = buffer_size

        self._ctx = mp.get_context("spawn")
        self._manager: T.Any | None = None
        self._shared_buffer: T.Any | None = None
        self._buffer_lock: T.Any | None = None
        self._error_state: T.Any | None = None
        self._ready_event: T.Any | None = None
        self._stop_event: T.Any | None = None
        self._worker: BaseProcess | None = None
        self._closed = False

    @staticmethod
    def _uniform_indices(buffer_length: int, segment_length: int) -> np.ndarray:
        if buffer_length < segment_length:
            raise ValueError(
                f"Cannot sample {segment_length} frames from a buffer with only {buffer_length} frames."
            )
        if segment_length == 1:
            return np.array([buffer_length - 1], dtype=np.int64)

        indices = np.floor(
            np.arange(segment_length, dtype=np.float64) * (buffer_length - 1) / (segment_length - 1)
        ).astype(np.int64)
        if np.unique(indices).size != segment_length:
            raise RuntimeError(
                f"Uniform sampling produced duplicate indices {indices.tolist()} for buffer_length={buffer_length}."
            )
        return indices

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray) -> torch.Tensor:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 frame, got shape {tuple(frame.shape)}")

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CameraStreamDataset has been closed and cannot be iterated again.")

    def _raise_if_worker_failed(self) -> None:
        if self._error_state is None:
            return

        message = self._error_state.get("message")
        if message is None:
            return

        self.close()
        raise RuntimeError(T.cast(str, message))

    def _ensure_worker_started(self) -> None:
        self._ensure_open()

        if self._worker is not None:
            self._raise_if_worker_failed()
            if not self._worker.is_alive():
                self.close()
                raise RuntimeError(f"Camera worker for camera id {self._cam_id} exited unexpectedly.")
            return

        manager = self._ctx.Manager()
        shared_buffer = manager.list()
        buffer_lock = self._ctx.Lock()
        error_state = manager.dict({"message": None})
        ready_event = self._ctx.Event()
        stop_event = self._ctx.Event()
        worker = self._ctx.Process(
            target=_camera_worker,
            args=(
                self._cam_id,
                self._freq_hz,
                self._buffer_size,
                shared_buffer,
                buffer_lock,
                error_state,
                ready_event,
                stop_event,
            ),
            daemon=True,
        )

        self._manager = manager
        self._shared_buffer = shared_buffer
        self._buffer_lock = buffer_lock
        self._error_state = error_state
        self._ready_event = ready_event
        self._stop_event = stop_event
        self._worker = worker

        try:
            worker.start()
            if not ready_event.wait(timeout=self._READY_TIMEOUT_SEC):
                self.close()
                raise TimeoutError(f"Timed out while starting camera id {self._cam_id}.")
            self._raise_if_worker_failed()
            if not worker.is_alive():
                self.close()
                raise RuntimeError(f"Camera worker for camera id {self._cam_id} exited before streaming.")
        except Exception:
            self.close()
            raise

    def _snapshot_and_clear_buffer(self) -> list[np.ndarray]:
        self._ensure_worker_started()
        assert self._shared_buffer is not None
        assert self._buffer_lock is not None

        while True:
            self._raise_if_worker_failed()
            if self._worker is None or not self._worker.is_alive():
                self.close()
                raise RuntimeError(f"Camera worker for camera id {self._cam_id} exited unexpectedly.")

            with self._buffer_lock:
                if len(self._shared_buffer) >= self._segment_length:
                    frames = list(self._shared_buffer)
                    while len(self._shared_buffer) > 0:
                        self._shared_buffer.pop()
                    return frames

            time.sleep(self._POLL_INTERVAL_SEC)

    def close(self) -> None:
        if self._closed:
            return

        worker = self._worker
        stop_event = self._stop_event
        manager = self._manager

        self._closed = True
        self._worker = None
        self._stop_event = None
        self._ready_event = None
        self._error_state = None
        self._shared_buffer = None
        self._buffer_lock = None
        self._manager = None

        if stop_event is not None:
            stop_event.set()

        if worker is not None:
            worker.join(timeout=self._JOIN_TIMEOUT_SEC)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=self._JOIN_TIMEOUT_SEC)

        if manager is not None:
            manager.shutdown()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __iter__(self) -> T.Iterator[MultiViewTask]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            raise RuntimeError(
                "CameraStreamDataset does not support DataLoader worker processes; use num_workers=0."
            )

        while True:
            frames = self._snapshot_and_clear_buffer()
            indices = self._uniform_indices(len(frames), self._segment_length)
            images = torch.stack([self._frame_to_tensor(frames[idx]) for idx in indices]).unsqueeze(0)
            yield MultiViewTask(
                multiview_input=MultiViewInput(images=images, intrinsics=None),
                scene_geometry=None,
            )
