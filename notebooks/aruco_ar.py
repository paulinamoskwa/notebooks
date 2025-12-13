import os
import cv2
import numpy as np
import pyrender
import trimesh

os.environ.pop("PYOPENGL_PLATFORM", None)

# The model is @ https://github.com/paulinamoskwa/detect-pikachu/tree/main/3d_modeling
MODEL_PATH = "model_3d/modelpika_updated.glb"


def generate_aruco_marker() -> None:
    """
    Generate a 4x4 ArUco marker image and save it to a file.
    This has to be done only once, then the image can be used for detection.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, 0, 400)
    cv2.imwrite("aruco_marker.png", marker_img)
    print("Saved aruco_marker.png")


def load_model(mesh_path: str, target_marker_size: float = 0.05) -> pyrender.Mesh:
    """
    Load a 3D mesh (GLB) from a file, process it to be properly oriented, centered,
    and scaled, and convert it to a pyrender mesh ready for rendering.
    """
    mesh = trimesh.load(mesh_path)

    # If the file contains a scene with multiple meshes, merge them into a single mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Ensure the mesh has vertex colors if it has any visual information
    if hasattr(mesh, "visual") and mesh.visual.kind != "vertex":
        mesh.visual = mesh.visual.to_color()

    # Pre-rotate the mesh upright (rotate around X axis by 90 degrees)
    # This is because the model is facing the wrong way
    rot = cv2.Rodrigues(np.array([np.pi / 2, 0, 0]))[0]  # rotation matrix
    mesh.apply_transform(np.vstack([np.hstack([rot, np.zeros((3,1))]), [0,0,0,1]]))

    # Compute mesh bounds for centering
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]

    # Center mesh in X/Y axes, keep Z at the base (so mesh sits on Z=0)
    center = [(bounds[0][0]+bounds[1][0])/2, (bounds[0][1]+bounds[1][1])/2, bounds[0][2]]
    mesh.apply_translation(-np.array(center))

    # Scale the mesh so its largest dimension matches the target marker size
    scale_factor = target_marker_size / mesh.extents.max()
    mesh.apply_scale(scale_factor)
    
    # Convert the processed Trimesh object into a Pyrender mesh (disable smoothing)
    return pyrender.Mesh.from_trimesh(mesh, smooth=False)


def build_camera(fx: float, fy: float, cx: float, cy: float) -> pyrender.IntrinsicsCamera:
    """Create a pyrender intrinsics camera that matches the live frame."""
    return pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=3.0)


def rvec_tvec_to_pose(rvec, tvec) -> np.ndarray:
    """Convert OpenCV rvec (rotation vector) and tvec (translation vector) 
    to a 4x4 OpenGL-compatible pose matrix."""

    # Convert rotation vector to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Build 4x4 homogeneous transformation
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R
    pose[:3, 3] = tvec.flatten()

    # OpenCV to OpenGL coordinate conversion
    cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float32)
    return cv_to_gl @ pose


def main():
    """
    - Opens camera
    - Detects ArUco markers
    - Renders a 3D model on top of detected markers in real-time using pyrender
    """
    marker_length_m = 0.05
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    model_mesh = load_model(MODEL_PATH, target_marker_size=marker_length_m*1.5)

    cap = cv2.VideoCapture(0)

    renderer = None
    renderer_size = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Approximate camera intrinsics using image size
        fx = fy = float(w)
        cx = w / 2.0
        cy = h / 2.0
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        if renderer is None or renderer_size != (w, h):
            if renderer is not None:
                renderer.delete()
            renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
            renderer_size = (w, h)

        # Detect ArUco markers
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Estimate pose of first detected marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length_m, camera_matrix, dist_coeffs
            )

            rvec, tvec = rvecs[0], tvecs[0]

            # Draw marker outlines and axes on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # Build Pyrender scene
            scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.4,0.4,0.4,1.0])
            cam = build_camera(fx, fy, cx, cy)
            scene.add(cam, pose=np.eye(4))
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
            scene.add(light, pose=np.eye(4))

            # Convert marker pose to OpenGL coordinates
            marker_pose = rvec_tvec_to_pose(rvec, tvec)

            # Optional lift matrix if model should sit slightly above marker
            lift = np.eye(4, dtype=np.float32)
            lift[2,3] = 0.0

            scene.add(model_mesh, pose=marker_pose @ lift)

            # Render scene to offscreen buffer
            color_rgba, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.FLAT)
            overlay_rgb = color_rgba[:, :, :3][:, :, ::-1]  # BGR
            alpha = color_rgba[:, :, 3:4].astype(np.float32)/255.0

            # Alpha blend rendered model onto camera frame
            frame = (frame.astype(np.float32)*(1-alpha) + overlay_rgb.astype(np.float32)*alpha).astype(np.uint8)

        cv2.imshow("AR Model", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    renderer.delete()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
