"""Minimal VGGT-SLAM wrapper.

Encapsulates the VGGT-SLAM pipeline into a single streaming class.
Accepts single-frame MultiViewInput instances, runs keyframe selection,
VGGT inference, SL(4) pose graph optimization with loop closure, and
returns global SceneGeometry on demand.
"""

import numpy as np
import cv2
import torch
import pypose as pp
from scipy.spatial.transform import Rotation

from ....utility.diagnostic import Diagnostics
from ....interface.geometric_model import (
    MultiViewInput,
    SceneGeometry,
    PoseConvention,
    GeometricPredictorLike,
)
from .frame_overlap import FrameTracker
from .map import GraphMap
from .submap import Submap
from .graph import PoseGraph
from .loop_closure import ImageRetrieval
from .scale_solver import estimate_scale_pairwise

from .utility import closed_form_inverse_se3, depth_to_cam_coords_points


class VGGTSlam:
    """Streaming VGGT-SLAM with SL(4) pose graph backend.

    Usage::

        slam = VGGTSlam(model)
        for frame in frames:
            slam.process_frame(frame)
        slam.finalize()
        scene = slam.get_scene()
    """

    def __init__(
        self,
        model: GeometricPredictorLike,
        *,
        submap_size: int = 16,
        overlapping_window_size: int = 1,
        max_loops: int = 1,
        min_disparity: float = 50.0,
        conf_threshold: float = 25.0,
        lc_thres: float = 0.95,
    ):
        self._model = model
        self._submap_size = submap_size
        self._overlapping_window_size = overlapping_window_size
        self._max_loops = max_loops
        self._min_disparity = min_disparity
        self._conf_threshold = conf_threshold
        self._lc_thres = lc_thres

        self._flow_tracker = FrameTracker()
        self._map = GraphMap()
        self._graph = PoseGraph()
        self._image_retrieval = ImageRetrieval()

        self._current_submap: Submap | None = None
        self._pending_frames: list[MultiViewInput] = []
        self._frame_counter: int = 0

        self._submap_inputs: dict[int, list[MultiViewInput]] = {}
        self._submap_depths: dict[int, np.ndarray] = {}
        self._submap_depths_conf: dict[int, np.ndarray] = {}
        self._submap_infer_masks: dict[int, np.ndarray] = {}

    # ==== Public API ====

    def process_frame(self, frame: MultiViewInput) -> None:
        """Ingest one or more frames. ``frame.images`` must have shape ``[1, S, 3, H, W]`` with S >= 1."""
        B, S = frame.images.shape[:2]
        if B != 1:
            raise ValueError(f"Expected batch size 1, got {B}")
        if S < 1:
            raise ValueError(f"Expected at least 1 frame (S >= 1), got S={S}")

        for i in range(S):
            single_image = frame.images[:, i : i + 1]
            single_intrinsics = (
                frame.intrinsics[:, i : i + 1] if frame.intrinsics is not None else None
            )
            single_frame = MultiViewInput(images=single_image, intrinsics=single_intrinsics)

            img_np = (single_image[0, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            if self._flow_tracker.compute_disparity(img_bgr, self._min_disparity):
                self._pending_frames.append(single_frame)

            if len(self._pending_frames) >= self._submap_size + self._overlapping_window_size:
                self._process_submap()

    def finalize(self) -> None:
        """Flush any remaining buffered frames through the pipeline."""
        if self._pending_frames:
            self._process_submap()

    def get_scene(self) -> tuple[MultiViewInput, SceneGeometry]:
        """Return the current global reconstruction."""
        if self._map.get_num_submaps() == 0:
            raise RuntimeError("No submaps have been processed yet")

        all_inputs: list[MultiViewInput] = []
        all_poses: list[np.ndarray] = []
        all_points: list[np.ndarray] = []
        all_depths: list[np.ndarray] = []
        all_depths_conf: list[np.ndarray] = []
        all_intrinsics: list[np.ndarray] = []
        all_infer_masks: list[np.ndarray] = []

        for submap in self._map.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue

            sid = submap.get_id()
            n_frames = len(submap.get_all_poses())

            all_inputs.extend(self._submap_inputs[sid])

            poses_world = submap.get_all_poses_world(self._graph)
            for pose_4x4 in poses_world:
                R = pose_4x4[:3, :3]
                t = pose_4x4[:3, 3]
                quat = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
                all_poses.append(np.concatenate([t, quat]).astype(np.float32))

            point_list, _, _ = submap.get_points_list_in_world_frame(self._graph)
            for pts_hwc in point_list:
                all_points.append(pts_hwc.transpose(2, 0, 1).astype(np.float32))

            if sid in self._submap_depths:
                for i in range(n_frames):
                    all_depths.append(self._submap_depths[sid][i][np.newaxis].astype(np.float32))

            if sid in self._submap_depths_conf:
                for i in range(n_frames):
                    all_depths_conf.append(
                        self._submap_depths_conf[sid][i][np.newaxis].astype(np.float32)
                    )

            if sid in self._submap_infer_masks:
                for i in range(n_frames):
                    all_infer_masks.append(self._submap_infer_masks[sid][i])

            for i in range(n_frames):
                all_intrinsics.append(submap.proj_mats[i][:3, :3].astype(np.float32))

        mv_input = MultiViewInput.collate(all_inputs)

        poses_t = torch.from_numpy(np.stack(all_poses)).unsqueeze(0)
        points_t = torch.from_numpy(np.stack(all_points)).unsqueeze(0)
        intrinsics_t = torch.from_numpy(np.stack(all_intrinsics)).unsqueeze(0)
        depths_t = torch.from_numpy(np.stack(all_depths)).unsqueeze(0) if all_depths else None
        depths_conf_t = (
            torch.from_numpy(np.stack(all_depths_conf)).unsqueeze(0) if all_depths_conf else None
        )

        infer_mask_t = (
            torch.from_numpy(np.stack(all_infer_masks)).unsqueeze(0)
            if all_infer_masks else None
        )

        scene = SceneGeometry(
            pose_convention=(PoseConvention.R, PoseConvention.D, PoseConvention.F),
            depths=depths_t,
            depths_conf=depths_conf_t,
            points=points_t,
            points_conf=None,
            poses=poses_t,
            intrinsics=intrinsics_t,
            infer_mask=infer_mask_t,
        )
        return mv_input, scene

    # ==== Internal pipeline ====

    def _process_submap(self) -> None:
        """Run VGGT on buffered frames, add submap to map, and optimize."""
        images = torch.cat([f.images for f in self._pending_frames], dim=1)  # [1, S, 3, H, W]
        batch_input = MultiViewInput(images=images, intrinsics=None)

        S = images.shape[1]

        # Assign submap ID
        if self._map.get_largest_key() is None:
            new_id = 0
        else:
            new_id = (
                self._map.get_largest_key()
                + self._map.get_latest_submap().get_last_non_loop_frame_index()
                + 1
            )

        self._submap_inputs[new_id] = list(self._pending_frames)

        # Create submap skeleton (before adding to map so retrieval sees only prior submaps)
        new_submap = Submap(new_id)
        new_submap.add_all_frames(images[0])  # [S, 3, H, W] tensor for SALAD

        frame_start = self._frame_counter
        new_submap.frame_ids = [float(frame_start + i) for i in range(S)]
        self._frame_counter += S

        img_names = [f"frame_{frame_start + i}" for i in range(S)]
        new_submap.set_last_non_loop_frame_index(S - 1)
        new_submap.set_img_names(img_names)

        new_submap.set_all_retrieval_vectors(
            self._image_retrieval.get_all_submap_embeddings(new_submap)
        )
        self._current_submap = new_submap

        # VGGT inference
        with torch.inference_mode():
            result = self._model(batch_input)

        predictions = self._bridge_scene_geometry(result, images)

        # Loop closure detection (new submap not in map yet)
        detected_loops = self._image_retrieval.find_loop_closures(
            self._map, new_submap,
            max_loop_closures=self._max_loops,
            max_similarity_thres=self._lc_thres,
        )
        predictions["detected_loops"] = detected_loops

        if len(detected_loops) > 0:
            Diagnostics.log(f"detected_loops {detected_loops}")
            retrieved_frames = self._map.get_frames_from_loops(detected_loops)

            query_frame = new_submap.get_frame_at_index(detected_loops[0].query_submap_frame)
            lc_frames = torch.stack([query_frame, retrieved_frames[0]], dim=0)  # [2, 3, H, W]

            with torch.inference_mode():
                lc_input = MultiViewInput(images=lc_frames.unsqueeze(0), intrinsics=None)
                lc_result = self._model(lc_input)

            self._graph.increment_loop_closure()
            lc_preds = self._bridge_scene_geometry(lc_result, lc_frames.unsqueeze(0))
            predictions["extrinsic_lc"] = lc_preds["extrinsic"]
            predictions["intrinsic_lc"] = lc_preds["intrinsic"]
            predictions["depth_lc"] = lc_preds["depth"]
            predictions["depth_conf_lc"] = lc_preds["depth_conf"]
            predictions["frames_lc"] = lc_frames
            predictions["frames_lc_names"] = [
                new_submap.get_img_names_at_index(detected_loops[0].query_submap_frame),
                self._map.get_submap(detected_loops[0].detected_submap_id)
                    .get_img_names_at_index(detected_loops[0].detected_submap_frame),
            ]

        self._add_points(predictions)
        self._graph.optimize()

        self._pending_frames = self._pending_frames[-self._overlapping_window_size:]

    def _bridge_scene_geometry(
        self, result: SceneGeometry, images: torch.Tensor
    ) -> dict:
        """Convert a VGGT SceneGeometry into the numpy dict consumed by the SL4 pipeline.

        Returns dict with keys: extrinsic, intrinsic, depth, depth_conf, images.
        All numpy, batch dimension squeezed.
        """
        cam_to_world_4x4 = pp.SE3(result.poses[0]).matrix().float().cpu().numpy()
        world_to_cam_4x4 = np.linalg.inv(cam_to_world_4x4)
        extrinsics = world_to_cam_4x4[:, :3, :]

        intrinsics = result.intrinsics[0].float().cpu().numpy()
        depths = result.depths[0].permute(0, 2, 3, 1).float().cpu().numpy()  # [S, H, W, 1]
        depths_conf = result.depths_conf[0, :, 0].float().cpu().numpy()  # [S, H, W]
        images_np = images[0].float().cpu().numpy()  # [S, 3, H, W]

        infer_mask = (
            result.infer_mask[0].cpu().numpy() if result.infer_mask is not None else None
        )

        return {
            "extrinsic": extrinsics,
            "intrinsic": intrinsics,
            "depth": depths,
            "depth_conf": depths_conf,
            "images": images_np,
            "infer_mask": infer_mask,
        }

    def _add_points(self, pred_dict: dict) -> None:
        """Populate the current submap, add it to the map, and wire up graph edges."""
        images = pred_dict["images"]            # (S, 3, H, W) float [0,1]
        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        detected_loops = pred_dict["detected_loops"]
        depth_map = pred_dict["depth"]            # (S, H, W, 1)
        conf = pred_dict["depth_conf"]            # (S, H, W)

        cam_points = np.stack([
            depth_to_cam_coords_points(depth_map[i].squeeze(-1), intrinsics_cam[i])
            for i in range(depth_map.shape[0])
        ], axis=0)
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # (S, H, W, 3)
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)

        N = cam_to_world.shape[0]
        K_4x4 = np.tile(np.eye(4), (N, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam
        world_to_cam = np.linalg.inv(cam_to_world)

        submap_id_prev = self._map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self._current_submap.get_id()
        frame_id_prev = None

        if submap_id_prev is not None:
            frame_id_prev = self._map.get_latest_submap(
                ignore_loop_closure_submaps=True
            ).get_last_non_loop_frame_index()

        self._current_submap.add_all_poses(world_to_cam)
        self._current_submap.add_all_points(
            cam_points, colors, conf, self._conf_threshold, K_4x4
        )
        self._current_submap.set_conf_masks(conf)
        self._map.add_submap(self._current_submap)

        self._submap_depths[submap_id_curr] = depth_map.squeeze(-1)
        self._submap_depths_conf[submap_id_curr] = conf
        if pred_dict["infer_mask"] is not None:
            self._submap_infer_masks[submap_id_curr] = pred_dict["infer_mask"]

        self._add_edge(submap_id_curr, 0, submap_id_prev, frame_id_prev, is_loop_closure=False)

        for loop in detected_loops:
            assert loop.query_submap_id == self._current_submap.get_id()

            cam_to_world_lc = closed_form_inverse_se3(pred_dict["extrinsic_lc"])
            K_4x4_lc = np.tile(np.eye(4), (2, 1, 1))
            K_4x4_lc[:, :3, :3] = pred_dict["intrinsic_lc"]
            world_to_cam_lc = np.linalg.inv(cam_to_world_lc)
            depth_map_lc = pred_dict["depth_lc"]
            conf_lc = pred_dict["depth_conf_lc"]

            cam_points_lc = np.stack([
                depth_to_cam_coords_points(depth_map_lc[i].squeeze(-1), pred_dict["intrinsic_lc"][i])
                for i in range(depth_map_lc.shape[0])
            ], axis=0)

            lc_submap_num = (
                self._map.get_largest_key()
                + self._map.get_latest_submap().get_last_non_loop_frame_index()
                + 1
            )

            lc_submap = Submap(lc_submap_num)
            lc_submap.set_lc_status(True)
            lc_submap.add_all_frames(pred_dict["frames_lc"])
            lc_submap.set_frame_ids(pred_dict["frames_lc_names"])
            lc_submap.set_last_non_loop_frame_index(1)

            lc_submap.add_all_poses(world_to_cam_lc)
            lc_colors = (
                pred_dict["frames_lc"].permute(0, 2, 3, 1).cpu().numpy() * 255
            ).astype(np.uint8)
            lc_submap.add_all_points(
                cam_points_lc, lc_colors, conf_lc, self._conf_threshold, K_4x4_lc
            )
            lc_submap.set_conf_masks(conf_lc)
            self._map.add_submap(lc_submap)

            self._add_edge(
                lc_submap_num, 0,
                loop.query_submap_id, loop.query_submap_frame,
                is_loop_closure=False,
            )
            self._add_edge(
                loop.detected_submap_id, loop.detected_submap_frame,
                lc_submap_num, 1,
                is_loop_closure=True,
            )

    def _add_edge(
        self,
        submap_id_curr: int,
        frame_id_curr: int,
        submap_id_prev: int | None = None,
        frame_id_prev: int | None = None,
        is_loop_closure: bool = False,
    ) -> None:
        """Add SL4 nodes and between-factors for inter/intra submap constraints."""
        if is_loop_closure and submap_id_prev is None:
            raise ValueError("Loop closure edge requires a previous submap")

        current_submap = self._map.get_submap(submap_id_curr)
        H_w_submap = np.eye(4)

        if submap_id_prev is not None:
            overlapping_node_id = submap_id_prev + frame_id_prev
            prior_submap = self._map.get_submap(submap_id_prev)

            current_conf = current_submap.get_conf_masks_frame(frame_id_curr)
            prior_conf = prior_submap.get_conf_masks_frame(frame_id_prev)
            good_mask = (
                (prior_conf > prior_submap.get_conf_threshold())
                * (current_conf > prior_submap.get_conf_threshold())
            ).reshape(-1)

            if np.sum(good_mask) < 100:
                Diagnostics.log(
                    "Not enough overlapping points, using less restrictive mask"
                )
                good_mask = (prior_conf > prior_submap.get_conf_threshold()).reshape(-1)
                if np.sum(good_mask) < 100:
                    good_mask = (prior_conf > 0).reshape(-1)

            P_temp = np.linalg.inv(prior_submap.proj_mats[-1]) @ current_submap.proj_mats[0]
            t1 = (
                P_temp[0:3, 0:3]
                @ current_submap.get_frame_pointcloud(frame_id_curr).reshape(-1, 3)[good_mask].T
            ).T
            t2 = prior_submap.get_frame_pointcloud(frame_id_prev).reshape(-1, 3)[good_mask]

            scale_factor = estimate_scale_pairwise(t1, t2)[0]
            Diagnostics.log(f"scale factor {scale_factor}")
            H_scale = np.diag((scale_factor, scale_factor, scale_factor, 1.0))

            H_overlap = (
                np.linalg.inv(prior_submap.proj_mats[-1])
                @ current_submap.proj_mats[0]
                @ H_scale
            )
            H_w_submap = self._graph.get_homography(overlapping_node_id) @ H_overlap

            if not is_loop_closure:
                self._graph.add_homography(submap_id_curr + frame_id_curr, H_w_submap)

            self._graph.add_between_factor(
                overlapping_node_id,
                submap_id_curr + frame_id_curr,
                H_overlap,
                self._graph.intra_submap_noise,
            )
        else:
            if submap_id_curr != 0 or frame_id_curr != 0:
                raise ValueError("First node must be submap 0, frame 0")
            self._graph.add_homography(0, H_w_submap)
            self._graph.add_prior_factor(0, H_w_submap)

        if is_loop_closure:
            return

        world_to_cam = current_submap.get_all_poses()
        for index in range(1, len(world_to_cam)):
            H_inner = world_to_cam[index - 1] @ np.linalg.inv(world_to_cam[index])
            current_node = self._graph.get_homography(submap_id_curr + index - 1) @ H_inner

            self._graph.add_homography(submap_id_curr + index, current_node)
            self._graph.add_between_factor(
                submap_id_curr + index - 1,
                submap_id_curr + index,
                H_inner,
                self._graph.inner_submap_noise,
            )
