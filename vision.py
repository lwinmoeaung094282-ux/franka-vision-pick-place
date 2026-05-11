"""
vision.py — Camera-based object detection for Isaac Sim 5.1

IMPORTANT: This module imports omni.replicator which requires SimulationApp
to already be running. Never import this at the top level of run.py —
always import AFTER SimulationApp is created.

Detection strategy:
  1. Attach an RGB + semantic_segmentation annotator to a replicator camera.
  2. Find the pixels whose semantic label matches the target object label.
  3. Compute the 2-D centroid of that mask.
  4. Cast a ray from the camera eye through that pixel and intersect it with
     the horizontal table plane (z = plane_z) to recover the 3-D world XY.
     This is simpler and more robust than depth-buffer unprojection.
"""

import numpy as np


class VisionDetector:
    """
    Parameters
    ----------
    cam_eye    : (x, y, z) camera position in world space
    cam_target : (x, y, z) point the camera looks at
    cam_up     : up vector, default (0, 0, 1)
    fov_deg    : vertical FOV in degrees (default 60)
    resolution : (width, height) in pixels (default 256x256)
    label      : semantic label string attached to the pick object
    """

    def __init__(
        self,
        cam_eye=(1.2, 0.0, 1.0),
        cam_target=(0.35, 0.0, 0.0),
        cam_up=(0, 0, 1),
        fov_deg=60.0,
        resolution=(256, 256),
        label="pick_cube",
    ):
        self.cam_eye    = np.array(cam_eye,    dtype=float)
        self.cam_target = np.array(cam_target, dtype=float)
        self.cam_up     = np.array(cam_up,     dtype=float)
        self.fov_deg    = fov_deg
        self.W, self.H  = resolution
        self.label      = label

        self._camera    = None
        self._rp        = None
        self._rgb_annot = None
        self._seg_annot = None
        self._ready     = False
        self._cam2world = None

    # ------------------------------------------------------------------
    # Setup — call once after world.reset()
    # ------------------------------------------------------------------

    def setup(self):
        import carb
        import omni.replicator.core as rep

        # Prevent annotators from auto-capturing every physics frame
        carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)

        # Match the physical camera FOV to fov_deg so _pixel_to_ray stays consistent.
        # Omniverse default horizontal_aperture = 20.955 mm; for a 1:1 render W==H,
        # vertical_aperture == horizontal_aperture, so both axes share the same FOV.
        _APERTURE_MM = 20.955
        focal_length_mm = (_APERTURE_MM / 2.0) / np.tan(np.radians(self.fov_deg / 2.0))

        self._camera = rep.create.camera(
            position=tuple(self.cam_eye.tolist()),
            look_at=tuple(self.cam_target.tolist()),
            focal_length=focal_length_mm,
            horizontal_aperture=_APERTURE_MM,
        )
        self._rp = rep.create.render_product(self._camera, (self.W, self.H))

        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._seg_annot = rep.AnnotatorRegistry.get_annotator(
            "semantic_segmentation", init_params={"colorize": False}
        )
        self._rgb_annot.attach(self._rp)
        self._seg_annot.attach(self._rp)

        # Warm up the replicator graph
        rep.orchestrator.preview()

        self._build_cam2world()
        self._ready = True

    # ------------------------------------------------------------------
    # Detection — returns 3-D world position or None
    # ------------------------------------------------------------------

    def detect(self, table_z=0.026, label=None):
        """
        Detect the pick object and return its (x, y, z) world position.

        Uses ray–plane intersection against a horizontal plane at height
        `table_z` (= object centre Z when resting on the table).
        `label` overrides self.label for multi-object scenes.
        Returns None if the label is not visible.
        """
        import omni.replicator.core as rep

        if not self._ready:
            raise RuntimeError("Call setup() before detect()")

        # Trigger a single render + annotator update
        rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)

        seg_data = self._seg_annot.get_data()
        if seg_data is None or "data" not in seg_data:
            return None

        # Find the integer ID for the requested semantic label
        id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
        target_id = self._find_label_id(id_to_labels, label=label)
        if target_id is None:
            return None

        mask = seg_data["data"] == target_id
        if not mask.any():
            return None

        # Image-space centroid
        ys, xs = np.where(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())

        # World-space ray through (cx, cy)
        ray = self._pixel_to_ray(cx, cy)

        # Ray–plane intersection: plane z = table_z
        denom = ray[2]
        if abs(denom) < 1e-6:
            return None
        t = (table_z - self.cam_eye[2]) / denom
        if t <= 0:
            return None

        world_pos = self.cam_eye + t * ray
        return world_pos

    def get_rgb(self):
        """Return current RGB frame as (H, W, 3) uint8 array, or None."""
        import omni.replicator.core as rep
        if not self._ready:
            return None
        rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        data = self._rgb_annot.get_data()
        if data is None:
            return None
        return np.array(data, dtype=np.uint8)[:, :, :3]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_label_id(self, id_to_labels, label=None):
        target = label if label is not None else self.label
        for k, v in id_to_labels.items():
            if target in str(v):
                try:
                    return int(k)
                except (ValueError, TypeError):
                    pass
        return None

    def _build_cam2world(self):
        """Build the 4x4 camera-to-world transform from eye/target/up."""
        eye    = self.cam_eye
        target = self.cam_target
        up     = self.cam_up

        forward = target - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up_real = np.cross(right, forward)

        # Columns: right, up, -forward (OpenGL/Replicator camera convention)
        self._cam2world = np.eye(4)
        self._cam2world[:3, 0] = right
        self._cam2world[:3, 1] = up_real
        self._cam2world[:3, 2] = -forward
        self._cam2world[:3, 3] = eye

    def _pixel_to_ray(self, px, py):
        """Return normalized world-space ray direction for pixel (px, py)."""
        # NDC: x in [-1, 1] left→right, y in [-1, 1] bottom→top
        ndc_x = (px - self.W / 2.0 + 0.5) / (self.W / 2.0)
        ndc_y = (self.H / 2.0 - py - 0.5) / (self.H / 2.0)

        # Camera-space direction (perspective divide with focal length)
        f      = 1.0 / np.tan(np.radians(self.fov_deg / 2.0))
        aspect = self.W / self.H
        ray_cam = np.array([ndc_x * aspect / f, ndc_y / f, -1.0])
        ray_cam /= np.linalg.norm(ray_cam)

        # Rotate into world space
        ray_world = self._cam2world[:3, :3] @ ray_cam
        return ray_world / np.linalg.norm(ray_world)
