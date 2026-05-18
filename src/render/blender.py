"""Blender-backed renderer for colored point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..interface.geometric_model import MultiViewInput, PoseConvention, SceneGeometry

_BLENDER_WORLD_CONVENTION: tuple[PoseConvention, PoseConvention, PoseConvention] = (
    PoseConvention.R, PoseConvention.F, PoseConvention.U
)
_COLOR_ATTRIBUTE_NAME = "point_color"
_POINT_OBJECT_NAME = "ScenePointCloud"
_POINT_MATERIAL_NAME = "PointCloudMaterial"


# ==== Configuration ====


@dataclass(kw_only=True)
class BlenderRendererConfiguration:
    point_radius: float = 0.01
    downsample_factor: int = 1
    emit_mix: float = 0.35
    emit_strength: float = 1.0


# ==== Blender helpers ====


def _require_blender():
    try:
        import bpy
        from mathutils import Vector
    except ImportError as exc:
        raise ImportError(
            "BlenderRenderer requires Blender's Python API (`bpy`). "
            "Run inference through the Blender container/runtime."
        ) from exc

    return bpy, Vector


def _reset_blender_scene(bpy) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.engine = "CYCLES"


def _configure_world(bpy) -> None:
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new(name="World")
        bpy.context.scene.world = world

    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is None:
        raise RuntimeError("Expected Blender world to provide a Background node.")

    background.inputs["Color"].default_value = (0.02, 0.02, 0.02, 1.0)
    background.inputs["Strength"].default_value = 0.15


def _create_point_material(bpy, config: BlenderRendererConfiguration):
    material = bpy.data.materials.new(name=_POINT_MATERIAL_NAME)
    material.use_nodes = True
    node_tree = material.node_tree
    node_tree.nodes.clear()

    output = node_tree.nodes.new(type="ShaderNodeOutputMaterial")
    attribute = node_tree.nodes.new(type="ShaderNodeAttribute")
    attribute.attribute_name = _COLOR_ATTRIBUTE_NAME

    principled = node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
    principled.inputs["Roughness"].default_value = 0.4

    emission = node_tree.nodes.new(type="ShaderNodeEmission")
    emission.inputs["Strength"].default_value = config.emit_strength

    mix = node_tree.nodes.new(type="ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = config.emit_mix

    attribute.location = (-700, 0)
    principled.location = (-350, 120)
    emission.location = (-350, -80)
    mix.location = (-80, 20)
    output.location = (180, 20)

    node_tree.links.new(attribute.outputs["Color"], principled.inputs["Base Color"])
    node_tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
    node_tree.links.new(principled.outputs["BSDF"], mix.inputs[1])
    node_tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def _create_point_object(
    bpy,
    positions: np.ndarray,
    colors: np.ndarray,
    material,
):
    mesh = bpy.data.meshes.new(name=_POINT_OBJECT_NAME)
    mesh.vertices.add(len(positions))
    mesh.vertices.foreach_set("co", positions.reshape(-1).tolist())
    mesh.update()

    color_attribute = mesh.color_attributes.new(
        name=_COLOR_ATTRIBUTE_NAME,
        type="FLOAT_COLOR",
        domain="POINT",
    )
    color_rgba = np.ones((colors.shape[0], 4), dtype=np.float32)
    color_rgba[:, :3] = colors
    color_attribute.data.foreach_set("color", color_rgba.reshape(-1).tolist())

    point_object = bpy.data.objects.new(_POINT_OBJECT_NAME, mesh)
    point_object.data.materials.append(material)
    bpy.context.scene.collection.objects.link(point_object)
    return point_object


def _attach_geometry_nodes(
    bpy,
    point_object,
    material,
    config: BlenderRendererConfiguration,
) -> None:
    modifier = point_object.modifiers.new(name="PointCloudGeometry", type="NODES")
    node_group = bpy.data.node_groups.new(name="PointCloudGeometry", type="GeometryNodeTree")
    modifier.node_group = node_group

    node_group.interface.new_socket(
        name="Geometry",
        in_out="INPUT",
        socket_type="NodeSocketGeometry",
    )
    node_group.interface.new_socket(
        name="Geometry",
        in_out="OUTPUT",
        socket_type="NodeSocketGeometry",
    )

    group_input = node_group.nodes.new(type="NodeGroupInput")
    mesh_to_points = node_group.nodes.new(type="GeometryNodeMeshToPoints")
    set_material = node_group.nodes.new(type="GeometryNodeSetMaterial")
    group_output = node_group.nodes.new(type="NodeGroupOutput")

    group_input.location = (-500, 0)
    mesh_to_points.location = (-180, 0)
    set_material.location = (120, 0)
    group_output.location = (380, 0)

    mesh_to_points.mode = "VERTICES"
    mesh_to_points.inputs["Radius"].default_value = config.point_radius
    set_material.inputs["Material"].default_value = material

    node_group.links.new(group_input.outputs["Geometry"], mesh_to_points.inputs["Mesh"])
    node_group.links.new(mesh_to_points.outputs["Points"], set_material.inputs["Geometry"])
    node_group.links.new(set_material.outputs["Geometry"], group_output.inputs["Geometry"])


def _frame_scene(bpy, Vector, positions: np.ndarray) -> None:
    minimum = positions.min(axis=0)
    maximum = positions.max(axis=0)
    center = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    span = float(max(extent.max(), 1e-3))

    camera_data = bpy.data.cameras.new(name="PointCloudCamera")
    camera_object = bpy.data.objects.new("PointCloudCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)

    camera_offset = np.array([2.0, -2.0, 1.5], dtype=np.float32) * span
    camera_object.location = Vector((center + camera_offset).tolist())
    view_direction = Vector(center.tolist()) - camera_object.location
    camera_object.rotation_euler = view_direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.lens = 45.0
    bpy.context.scene.camera = camera_object

    light_data = bpy.data.lights.new(name="PointCloudSun", type="SUN")
    light_data.energy = 2.5
    light_object = bpy.data.objects.new("PointCloudSun", light_data)
    bpy.context.scene.collection.objects.link(light_object)
    light_object.location = camera_object.location
    light_object.rotation_euler = camera_object.rotation_euler


# ==== Point cloud extraction ====


def _validate_config(config: BlenderRendererConfiguration) -> None:
    if config.point_radius <= 0.0:
        raise ValueError(f"point_radius must be positive, got {config.point_radius}")
    if config.downsample_factor < 1:
        raise ValueError(
            f"downsample_factor must be an integer >= 1, got {config.downsample_factor}"
        )
    if not 0.0 <= config.emit_mix <= 1.0:
        raise ValueError(f"emit_mix must lie in [0, 1], got {config.emit_mix}")
    if config.emit_strength <= 0.0:
        raise ValueError(f"emit_strength must be positive, got {config.emit_strength}")


def _validate_inputs(scene: SceneGeometry, input_mvs: MultiViewInput) -> None:
    if scene.points is None:
        raise ValueError("BlenderRenderer requires scene.points (got None).")

    if scene.points.ndim != 5:
        raise ValueError(
            "BlenderRenderer expects scene.points to have shape [B, S, 3, H, W], "
            f"got {tuple(scene.points.shape)}"
        )

    if input_mvs.images.ndim != 5:
        raise ValueError(
            "BlenderRenderer expects input.images to have shape [B, S, 3, H, W], "
            f"got {tuple(input_mvs.images.shape)}"
        )

    if scene.points.shape[:2] != input_mvs.images.shape[:2]:
        raise ValueError(
            "Batch/view dimensions of scene.points and input.images must match, "
            f"got {tuple(scene.points.shape[:2])} and {tuple(input_mvs.images.shape[:2])}"
        )

    if scene.points.shape[-2:] != input_mvs.images.shape[-2:]:
        raise ValueError(
            "Spatial dimensions of scene.points and input.images must match, "
            f"got {tuple(scene.points.shape[-2:])} and {tuple(input_mvs.images.shape[-2:])}"
        )

    if scene.points.shape[2] != 3 or input_mvs.images.shape[2] != 3:
        raise ValueError(
            "BlenderRenderer expects 3-channel points/images, "
            f"got channels={scene.points.shape[2]} and {input_mvs.images.shape[2]}"
        )


def _downsample_spatial(tensor: torch.Tensor | None, factor: int) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[..., ::factor, ::factor]


def _extract_batch_points(
    scene: SceneGeometry,
    input_mvs: MultiViewInput,
    batch_index: int,
    downsample_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    if scene.points is None:
        raise ValueError("SceneGeometry.points is required before point extraction.")

    points = _downsample_spatial(scene.points[batch_index], downsample_factor)
    colors = _downsample_spatial(input_mvs.images[batch_index], downsample_factor)
    conf = _downsample_spatial(
        scene.points_conf[batch_index] if scene.points_conf is not None else None,
        downsample_factor,
    )
    mask = _downsample_spatial(
        scene.infer_mask[batch_index] if scene.infer_mask is not None else None,
        downsample_factor,
    )

    if points is None or colors is None:
        raise RuntimeError("Point extraction requires both positions and colors to be present.")

    flat_points = points.permute(0, 2, 3, 1).reshape(-1, 3)
    flat_colors = colors.permute(0, 2, 3, 1).reshape(-1, 3)

    keep = torch.isfinite(flat_points).all(dim=1) & torch.isfinite(flat_colors).all(dim=1)
    if conf is not None:
        keep &= conf.reshape(-1) > 0
    if mask is not None:
        keep &= ~mask.reshape(-1)

    if not bool(keep.any()):
        raise ValueError(
            f"BlenderRenderer found no valid points for batch_index={batch_index} after filtering."
        )

    kept_points = flat_points[keep].contiguous().numpy().astype(np.float32)
    kept_colors = flat_colors[keep].clamp(0.0, 1.0).contiguous().numpy().astype(np.float32)
    return kept_points, kept_colors


# ==== Renderer ====


class BlenderRenderer:
    def __init__(self, config: BlenderRendererConfiguration | None = None) -> None:
        self.config = config if config is not None else BlenderRendererConfiguration()
        _validate_config(self.config)

    def render(self, scene: SceneGeometry, input: MultiViewInput, save_to: str | Path) -> None:
        bpy, Vector = _require_blender()
        _validate_inputs(scene, input)

        scene.to(device="cpu", convention=_BLENDER_WORLD_CONVENTION)
        input.to(device="cpu")

        batch_size = input.images.shape[0]
        save_to = Path(save_to)
        save_to.parent.mkdir(parents=True, exist_ok=True)

        for batch_index in range(batch_size):
            positions, colors = _extract_batch_points(
                scene,
                input,
                batch_index,
                self.config.downsample_factor,
            )

            _reset_blender_scene(bpy)
            _configure_world(bpy)

            material = _create_point_material(bpy, self.config)
            point_object = _create_point_object(bpy, positions, colors, material)
            _attach_geometry_nodes(bpy, point_object, material, self.config)
            _frame_scene(bpy, Vector, positions)

            output_path = save_to.parent / f"{save_to.name}_{batch_index:02d}.blend"
            bpy.ops.wm.save_as_mainfile(
                filepath=str(output_path),
                check_existing=False,
                compress=False,
            )
