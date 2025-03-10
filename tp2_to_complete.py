import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

# Load the image
image_path = "/path/to/png/image/frame1.png"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Open3D

# Load the LiDAR point cloud
lidar_path = "/path/to/binary/pointcloud/frame1.bin"  # Replace with your LiDAR .bin file path
lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # (x, y, z, intensity)

# Projection matrix (P2)
P2 = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
    [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
    [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
])  # 3x4 projection matrix

# LiDAR to Camera transformation matrix (Tr_velo_to_cam)
Tr_velo_to_cam = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]
])  # 3x4 matrix

### **Step 1: Project LiDAR Points onto Image Plane** ###
def project_lidar_to_image(lidar_points, Tr_velo_to_cam, P2):
    """Projects LiDAR points onto the image plane and returns 2D pixel coordinates."""
    
    # TODO: Convert LiDAR points to homogeneous coordinates
    lidar_hom =  # <-- Complete this part

    # Transform points to camera coordinates
    cam_coords =  # <-- Complete this part

    # TODO: Filter out points behind the camera
    cam_coords =  # <-- Complete this part

    # Convert to homogeneous coordinates for projection
    cam_hom = np.hstack((cam_coords, np.ones((cam_coords.shape[0], 1))))

    # Project onto image plane
    img_coords = (P2 @ cam_hom.T).T

    # TODO: Normalize homogeneous coordinates
    img_coords[:, 0] =  # <-- Complete this part
    img_coords[:, 1] =  # <-- Complete this part

    return img_coords, cam_coords

# Get 2D pixel coordinates and filtered LiDAR points
img_coords, cam_coords = project_lidar_to_image(lidar_points, Tr_velo_to_cam, P2)

### **Step 2: Extract Color from Image** ###
def get_image_colors(img_coords, image):
    """Extracts RGB colors from the image for each valid LiDAR point."""
    img_h, img_w, _ = image.shape
    
    # TODO: Keep only points within image boundaries
    valid_mask =  # <-- Complete this part
    img_coords = img_coords[valid_mask]
    cam_coords_filtered = cam_coords[valid_mask]
    
    # Convert coordinates to integer pixel indices
    img_x = img_coords[:, 0].astype(int)
    img_y = img_coords[:, 1].astype(int)

    # TODO: Extract RGB colors from image pixels
    colors =  # <-- Complete this part

    return cam_coords_filtered, colors

# Extract colors
colored_points, point_colors = get_image_colors(img_coords, image)

### **Step 3: Assign Colors to Point Cloud and Save as `.pcd`** ###
def save_colored_point_cloud_pcd(points, colors):
    """Saves the colored point cloud as a .pcd file using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # TODO: Save the point cloud as a .pcd file
    o3d.io.write_point_cloud( )  # <-- Complete this part
    print(f"Saved point cloud as colored_pointcloud.pcd")

    # Show the result
    o3d.visualization.draw_geometries([pcd], window_name="Colored LiDAR Point Cloud")

# Save and visualize
save_colored_point_cloud_pcd(colored_points, point_colors)
