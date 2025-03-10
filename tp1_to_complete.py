import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

# Load the LiDAR point cloud
# The file contains points in the format (x, y, z, intensity)
lidar_path = "/path/to/binary/pointcloud"  # Replace with actual path
lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # (x, y, z, intensity)

# Load the corresponding image
image_path = "/path/to/png"  # Replace with actual path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib visualization

# Projection matrix (P2)
# Used to project 3D points onto the 2D image plane
P2 = np.array([
    [7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
    [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
    [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]
])  # 3x4 projection matrix

# LiDAR to Camera transformation matrix from LiDAR coordinate system to camera coordinate system
Tr_velo_to_cam = np.array([
    [7.533745e-03, -9.999714e-01, -6.166020e-04, -4.069766e-03],
    [1.480249e-02, 7.280733e-04, -9.998902e-01, -7.631618e-02],
    [9.998621e-01, 7.523790e-03, 1.480755e-02, -2.717806e-01]
])  # 3x4 matrix

### **Step 1: Show LiDAR Point Cloud in Open3D (Black Points) ###
def visualize_point_cloud(lidar_points):
    """Visualizes the LiDAR point cloud using Open3D with black points."""
    pcd = o3d.geometry.PointCloud() 
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3]) # Use only (x, y, z) coordinates
    o3d.visualization.draw_geometries([pcd], window_name="LiDAR Point Cloud")

# Show raw LiDAR data first
visualize_point_cloud(lidar_points)

### **Step 2: Transform and Project LiDAR Points onto the Image using Open3D** ###
def project_lidar_to_image(lidar_points, Tr_velo_to_cam, P2):
    """Transforms LiDAR points using Open3D and projects them onto the image plane."""
    
    # TODO: Convert LiDAR points to homogeneous coordinates
    lidar_hom =  # <-- Complete this part

    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_hom[:, :3])
    
    # Transform points to the camera coordinate system
    cam_coords = (Tr_velo_to_cam @ lidar_hom.T).T #( T is the transpose )


    # TODO: Filter points that are behind the camera (negative z-values)
    cam_coords =  # <-- Complete this part

    # Convert to homogeneous coordinates for projection
    cam_hom = np.hstack((cam_coords, np.ones((cam_coords.shape[0], 1))))  # (N, 4)

    # Project onto the image plane
    img_coords = (P2 @ cam_hom.T).T  # (N, 3)

    # TODO: Normalize homogeneous coordinates
    img_coords[:, 0] =  # <-- Complete this part
    img_coords[:, 1] =  # <-- Complete this part

    return img_coords

# Get projected points
img_coords = project_lidar_to_image(lidar_points, Tr_velo_to_cam, P2)

### **Step 3: Overlay Projected Points on Image in Black** ###
# Image dimensions
img_h, img_w, _ = image.shape

# Keep points within image boundaries
valid_mask = (img_coords[:, 0] >= 0) & (img_coords[:, 0] < img_w) & (img_coords[:, 1] >= 0) & (img_coords[:, 1] < img_h)
img_coords = img_coords[valid_mask]

# TODO: Draw black points on image
for x, y in img_coords[:, :2]:
    cv2.circle(, (int(x), int(y)), 2,  , -1)  # <-- Complete this part

# Save the projected image
output_path = "projected_image_black.png"
cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Display the final image
plt.figure(figsize=(12, 6))
plt.imshow(image)
plt.axis("off")
plt.title("Projected LiDAR Points on Image (Black)")
plt.show()