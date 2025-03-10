import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

# Load the LiDAR point cloud
lidar_path = "path/to/binary/file"  # Replace with actual path
lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)  # (x, y, z, intensity)

### **Step 1: Show LiDAR Point Cloud in Open3D ###
def visualize_point_cloud(lidar_points):
    """Visualizes the LiDAR point cloud using Open3D with black points."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])

    colors = np.zeros((lidar_points.shape[0], 3))  # Create an (N,3) array filled with zeros (black color in RGB)
    pcd.colors = o3d.utility.Vector3dVector(colors)  # Assign black color to all points in the point cloud


    # Set all points to black
    o3d.visualization.draw_geometries([pcd], window_name="LiDAR Point Cloud")

# Show raw LiDAR data first
visualize_point_cloud(lidar_points)