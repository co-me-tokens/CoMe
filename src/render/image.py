"""
Render SceneGeometry + MultiViewInput into a matplotlib figure.

Layout per batch item b:

    vertical([3DPlot, Horizontal([Vertical([RGB, Depth], 1:1), ... x P])], 2:1)
"""

import numpy as np
import pypose as pp
import matplotlib.pyplot as plt
import jaxtyping as jt
from dataclasses import dataclass
from beartype import beartype
from pathlib import Path
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

from ..interface.geometric_model import SceneGeometry, MultiViewInput, PoseConvention

_RDF: tuple[PoseConvention, PoseConvention, PoseConvention] = (
    PoseConvention.R, PoseConvention.D, PoseConvention.F
)

# ==== Configuration ====

@jt.jaxtyped(typechecker=beartype)
@dataclass
class MatplotlibConfiguration:
    pose_axis_len: float = 0.15  # Pose axis length (unit: m)
    mask_alpha: float = 0.7


# ==== Helpers ====


def _blend_mask(image: np.ndarray, mask: np.ndarray, mask_alpha: float) -> np.ndarray:
    """Blend infer_mask as white directly into an RGB image.

    Args:
        image:      Float image of shape (H, W, 3) in [0, 1].
        mask:       Boolean array of shape (H, W).
        mask_alpha: White overlay strength in [0, 1].
    """
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"_blend_mask expects an RGB image of shape (H, W, 3), got {image.shape}")
    if mask.ndim != 2:
        raise ValueError(f"_blend_mask expects a 2D mask, got {mask.shape}")
    if image.shape[:2] != mask.shape:
        raise ValueError(
            f"_blend_mask expects image/mask spatial shapes to match, got {image.shape[:2]} and {mask.shape}"
        )
    if not (0.0 <= mask_alpha <= 1.0):
        raise ValueError(f"mask_alpha must lie in [0, 1], got {mask_alpha}")

    blended = image.copy()
    blended[mask] = blended[mask] * (1.0 - mask_alpha) + mask_alpha
    return blended


def _depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    """Convert a depth map to RGB using the same colormap normalization as imshow."""
    if depth.ndim != 2:
        raise ValueError(f"_depth_to_rgb expects a 2D depth map, got {depth.shape}")

    finite_mask = np.isfinite(depth)
    if not finite_mask.any():
        raise ValueError("Depth map contains no finite values.")

    norm = Normalize(
        vmin=float(depth[finite_mask].min()),
        vmax=float(depth[finite_mask].max()),
        clip=True,
    )
    return np.asarray(colormaps["Spectral"](norm(depth))[..., :3], dtype=np.float32)


def plot_poses(
    ax: Axes3D,
    scene: SceneGeometry,
    b: int,
    config: MatplotlibConfiguration,
) -> None:
    """Plot camera poses as RGB XYZ axes on a 3D Axes.

    Each camera is drawn as three quivers (X=red, Y=green, Z=blue) originating
    at the camera translation, pointing along the camera-local axes in the world
    frame.  Requires scene.poses to be set.

    Args:
        ax:     3D matplotlib axes.
        scene:  SceneGeometry (already moved to cpu / RDF convention by caller).
        b:      Batch index.
        config: Rendering configuration (provides pose_axis_len).
    """
    if scene.poses is None:
        raise ValueError("plot_poses requires scene.poses to be set.")

    poses = scene.poses  # (B, S, 7)
    S = poses.shape[1]
    L = config.pose_axis_len

    for s in range(S):
        t = poses[b, s, :3].numpy()               # (3,) translation
        R = pp.SO3(poses[b, s, 3:]).matrix().numpy()  # (3, 3) rotation, cols = camera axes in world

        for col, color in zip(range(3), ("red", "green", "blue")):
            axis = R[:, col] * L
            ax.quiver(
                t[0], t[1], t[2],
                axis[0], axis[1], axis[2],
                color=color, linewidth=1.0,
            )


def plot_depth(
    ax: Axes,
    scene: SceneGeometry,
    b: int,
    s: int,
    config: MatplotlibConfiguration,
) -> None:
    """Plot depth estimation for view (b, s) with optional infer_mask overlay.

    Args:
        ax:    2D matplotlib axes.
        scene: SceneGeometry (cpu).  scene.depths must be set.
        b:     Batch index.
        s:     Sequence index.
        config: Rendering configuration.
    """
    if scene.depths is None:
        raise ValueError("plot_depth requires scene.depths to be set.")

    depth_np = scene.depths[b, s, 0].numpy()  # (H, W)
    depth_rgb = _depth_to_rgb(depth_np)

    if scene.infer_mask is not None:
        depth_rgb = _blend_mask(depth_rgb, scene.infer_mask[b, s, 0].numpy(), config.mask_alpha)

    ax.imshow(depth_rgb)
    ax.axis("off")


def plot_image(
    ax: Axes,
    input: MultiViewInput,
    scene: SceneGeometry,
    b: int,
    s: int,
    config: MatplotlibConfiguration,
) -> None:
    """Plot the RGB image for view (b, s) with optional infer_mask overlay.

    Args:
        ax:    2D matplotlib axes.
        input: MultiViewInput (cpu).
        scene: SceneGeometry (cpu) — provides infer_mask.
        b:     Batch index.
        s:     Sequence index.
        config: Rendering configuration.
    """
    img_np = input.images[b, s].permute(1, 2, 0).numpy().clip(0.0, 1.0)  # (H, W, 3) float32

    if scene.infer_mask is not None:
        img_np = _blend_mask(img_np, scene.infer_mask[b, s, 0].numpy(), config.mask_alpha)

    ax.imshow(img_np)
    ax.axis("off")


# ==== Renderer ====

class MatplotlibRenderer:
    """Implements RendererLike: saves one PNG per batch item.

    Saves files named ``{save_to}_{b:02d}.png`` for b in 0..B-1.

    Figure structure (per batch item b):
        vertical([3DPlot, Horizontal([Vertical([RGB, Depth], 1:1), ... x P])], 2:1)
    """

    def __init__(self, config: MatplotlibConfiguration | None = None):
        self.config = config if config is not None else MatplotlibConfiguration()

    def render(
        self,
        scene:   SceneGeometry,
        input:   MultiViewInput,
        save_to: str | Path,
    ) -> None:
        """Generate one figure per batch item and save as PNG.

        Args:
            scene:   SceneGeometry with shape (B, P, ...).
            input:   MultiViewInput with shape (B, P, ...).
            save_to: Path prefix for output files (no extension).
        """
        scene.to(device="cpu", convention=_RDF)
        input.to(device="cpu")
        images = input.images

        B, P = images.shape[:2]
        H_img, W_img = images.shape[3], images.shape[4]
        save_to = Path(save_to)

        dpi = 200
        img_w_in  = W_img / dpi                 # one image width  in inches
        img_h_in  = H_img / dpi                 # one image height in inches
        title_h_in = 0.2                         # fixed height reserved for the frame title
        # Inner grid: top row = img + title, bottom row = img (depth)
        inner_w_in = P * img_w_in
        inner_h_in = (img_h_in + title_h_in) + img_h_in
        # 3-D plot section: same width, 1.5× the image section height (min 4 in)
        plot_h_in  = max(inner_h_in * 1.5, 4.0)
        fig_w = inner_w_in
        fig_h = plot_h_in + inner_h_in

        for b in range(B):
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

            outer = GridSpec(
                2, 1, figure=fig,
                height_ratios=[plot_h_in, inner_h_in],
                hspace=0.05,
            )
            ax3d: Axes3D = fig.add_subplot(outer[0], projection="3d")  # type: ignore[assignment]

            inner = GridSpecFromSubplotSpec(
                2, P, subplot_spec=outer[1],
                hspace=0.0, wspace=0.0,
                height_ratios=[img_h_in + title_h_in, img_h_in],
            )

            if scene.poses is not None:
                plot_poses(ax3d, scene, b, self.config)

            for s in range(P):
                ax_rgb = fig.add_subplot(inner[0, s])
                ax_dep = fig.add_subplot(inner[1, s])

                ax_rgb.set_title(f"Frame {s:02d}", fontsize=7, pad=2)
                plot_image(ax_rgb, input, scene, b, s, self.config)

                if scene.depths is not None:
                    plot_depth(ax_dep, scene, b, s, self.config)
                else:
                    ax_dep.axis("off")

            out_path = save_to.parent / f"{save_to.name}_{b:02d}.png"
            fig.savefig(out_path, bbox_inches="tight", dpi=150)
            plt.close(fig)
