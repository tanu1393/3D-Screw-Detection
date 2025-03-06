import cv2
import numpy as np
import open3d as o3d
import json
import os
import glob
import sys
import argparse
import logging
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_image(image_path):
    logging.info(f"Loading image from {image_path}")
    img = cv2.imread(image_path)
    return img


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding to enhance screw edges
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    # Morphological opening to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return processed


def visualize_2d_screw_detections(image, screws, output_2d_det_path):
    output = image.copy()
    for x, y in screws:
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)  # Draw red dots for screws
        cv2.putText(output, f"({x}, {y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 3, cv2.LINE_AA)  # Label with coordinates
    # Save the image
    cv2.imwrite(output_2d_det_path, output)


def detect_screws(image):
    processed = preprocess_image(image)
    circles = cv2.HoughCircles(
        processed, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=100,param2=40, minRadius=10, maxRadius=30
    )
    screws = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            mask = np.zeros_like(processed)
            cv2.circle(mask, (x, y), r, 255, -1)
            masked_img = cv2.bitwise_and(processed, processed, mask=mask)
            contours, _ = cv2.findContours(masked_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * (area / (perimeter * perimeter + 1e-6))
                if 0.75 < circularity < 1.2:
                    screws.append((x, y))
    logging.info(f"Detected {len(screws)} screws")
    return screws


def project_points(points_3d, intrinsics):
    points_2d = np.dot(points_3d, intrinsics.T)
    z_values = points_2d[:, 2]
    mask = np.abs(z_values) > 1e-6
    points_2d_projected = np.zeros_like(points_2d[:, :2]) # Initialize array for projected points
    points_2d_projected[mask] = points_2d[mask, :2] / z_values[mask, np.newaxis]
    return points_2d_projected


def find_3d_screw_positions(pointcloud_path, screws, intrinsics):
    logging.info(f"Loading point cloud from {pointcloud_path}")
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    points = np.asarray(pcd.points)
    screw_3d_positions = []
    for circle in screws:
        u, v = circle[0], circle[1]
        #Projects 3D points to the image plane, handling potential division by zero.
        projected_points = project_points(points, intrinsics)
        distances = np.linalg.norm(projected_points - np.array([u, v]), axis=1)
        valid_distances = distances[~np.isnan(distances)]
        if valid_distances.size > 0:
            nearest_idx = np.argmin(valid_distances)
            Z = points[nearest_idx, 2]  
        else:
            logging.warn("Warning: No valid points found for screw detection.")
            Z = 0
        X = (u - intrinsics[0, 2]) * Z / intrinsics[0, 0]
        Y = (v - intrinsics[1, 2]) * Z / intrinsics[1, 1]
        screw_3d_positions.append([X, Y, Z])
    logging.info(f"Computed 3D positions for {len(screw_3d_positions)} screws")
    return screw_3d_positions


def estimate_intrinsics():
    logging.info("Estimating camera intrinsics")
    focus_distance = 1100  # mm
    fov_x = 1090  # mm
    fov_y = 850   # mm
    image_width, image_height = 2448, 2048
    f_x = (image_width * focus_distance) / fov_x
    f_y = (image_height * focus_distance) / fov_y
    # Compute principal point (assuming center)
    c_x = image_width / 2
    c_y = image_height / 2
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    return K


def estimate_screw_orientation(pointcloud_path, screw_positions):
    logging.info(f"Loading point cloud from {pointcloud_path}")
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    point_cloud = np.asarray(pcd.points)
    orientations = []
    
    for pos in screw_positions:
        # Find neighbors within a 10mm radius
        neighbors = point_cloud[np.linalg.norm(point_cloud - pos, axis=1) < 10]

        # Check if enough neighbors exist for PCA. It requires at least 3 points
        if len(neighbors) < 3:
            logging.warning(f"Not enough neighbors found for screw at {pos}. Using default Z-axis orientation.")
            orientations.append(np.array([0, 0, 1]))  # Default orientation along Z-axis
            continue

        # Compute covariance matrix
        cov_matrix = np.cov(neighbors.T)
        # Handle invalid covariance matrix cases
        if np.isnan(cov_matrix).any() or np.isinf(cov_matrix).any():
            logging.warning(f"Covariance matrix contains NaN/Inf for screw at {pos}. Using default Z-axis orientation.")
            orientations.append(np.array([0, 0, 1]))
            continue

        # Perform PCA (Eigen decomposition)
        eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

        # Choose the eigenvector with the highest eigenvalue
        main_direction = eig_vecs[:, np.argmax(eig_vals)]
        orientations.append(main_direction)
    logging.info(f"Estimated orientations for {len(orientations)} screws.")
    return orientations


def transform_to_robot_frame(screw_positions, orientations, transform_matrix):
    logging.info("Transforming screw positions to robot coordinate system")
    screws_frame = []
    transform_matrix = np.array(transform_matrix)
    for pos, ori in zip(screw_positions, orientations):
        pos_homogeneous = np.append(pos, 1)
        pos_robot = transform_matrix @ pos_homogeneous
        ori_robot = transform_matrix[:3, :3] @ ori
        screws_frame.append({
            "position_robot": pos_robot[:3].tolist(),
            "orientation_robot": ori_robot.tolist(),
            "position_camera": pos,
            "orientation_camera": ori.tolist()
        })
    return screws_frame

def save_results(screws_frame, output_path):
    logging.info(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump({
            "screws": screws_frame
        }, f, indent=4)


def visualize_point_cloud_and_screws_open3d(pointcloud_path, screws_frame, output_file=None):
    logging.info(f"Loading point cloud from {pointcloud_path}")
    pcd = o3d.io.read_point_cloud(pointcloud_path)
    point_cloud = np.asarray(pcd.points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()
    if output_file:
        vis.create_window(visible=False)
    else:
        vis.create_window()

    try:
        # Add the point cloud to the visualizer
        vis.add_geometry(pcd)

        # Highlight detected screws in blue
        for index, screw in enumerate(screws_frame):
            pos = screw['position_camera']
            screw_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
            screw_sphere.paint_uniform_color([0, 0, 1])
            screw_sphere.translate(pos)
            vis.add_geometry(screw_sphere) 

        # Run the visualizer
        if output_file:
            # Capture the screen image and save it as a PNG file
            vis.poll_events()  # Poll events to ensure the visualizer is ready
            vis.update_renderer()  # Update the renderer
            vis.capture_screen_image(output_file)
            logging.info(f"Visualization saved to {output_file}")
        else:
            vis.run()

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
    finally:
        vis.destroy_window()

    #load the image file to add text
    if output_file:
        img = cv2.imread(output_file)

        # Define text properties
        text = "Screws are highlighted in Blue"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_color = (255, 0, 0)  # Blue color
        thickness = 3

        h, w, _ = img.shape
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = w - text_size[0] - 10
        text_y = text_size[1] + 10

        # Add a background rectangle and add text
        cv2.rectangle(img, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (255,255,255), -1)
        cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, thickness)
        
        #save image
        cv2.imwrite(output_file, img)

        # Convert the image to bytes
        is_success, buffer = cv2.imencode(".png", cv2.imread(output_file, cv2.IMREAD_COLOR))
        if is_success:
            return BytesIO(buffer)
    return None


def process_screw_detection(image_path, ply_path, json_path, output_path, output_2d_det_path, output_image_path=None):
    """Complete pipeline for screw detection and 3D pose estimation.
    -- load the image
    -- detect the screws
    -- estimate the intrinsic parameters for the camera
    -- calculate the 3d pose of the screws using the 2d detections
    -- calculate the orientation of the screws
    -- get the 3d screw poses in robot coordinate system
    -- Visualize the 3d screws in point cloud
    """
    try:
        image = load_image(image_path)
        screws_2d = detect_screws(image)
        with open(json_path, 'r') as f:
            transform_matrix = json.load(f)
        intrinsics = estimate_intrinsics()
        screw_3d_positions = find_3d_screw_positions(ply_path, screws_2d, intrinsics)
        orientations = estimate_screw_orientation(ply_path, screw_3d_positions)
        screws_frame = transform_to_robot_frame(screw_3d_positions, orientations, transform_matrix)
        visualize_2d_screw_detections(image, screws_2d, output_2d_det_path)
        save_results(screws_frame, output_path)
        img_bytes = visualize_point_cloud_and_screws_open3d(ply_path, screws_frame, output_image_path)
        return img_bytes
    except Exception as e:
        logging.error(f"Error processing screw detection: {e}", exc_info=True)
        return None
    
    
def get_file_paths(dataset_folder, output_folder):
    logging.info(f"Processing dataset folder: {dataset_folder}")
    image_path = glob.glob(os.path.join(dataset_folder, "*.png"))[0]
    ply_path = glob.glob(os.path.join(dataset_folder, "*.ply"))[0]
    json_path = glob.glob(os.path.join(dataset_folder, "*.json"))[0]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, base_name), exist_ok=True)
    output_json_path = os.path.join(output_folder, base_name, f"{base_name}_screw_positions.json")
    output_image_path = os.path.join(output_folder, base_name, f"{base_name}_visualization.png")
    output_2d_det_path = os.path.join(output_folder, base_name, f"{base_name}_2d_detections.png")
    return image_path, ply_path, json_path, output_json_path, output_2d_det_path, output_image_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process screw detection from images.")
    parser.add_argument("dataset_folder", help="Path to the dataset folder containing images.")
    parser.add_argument("--output_folder", default="results", help="Path to the output folder (default: output).")
    parser.add_argument("--visualize", action="store_true", help="Visualize the results (default: False).")
    parser.add_argument("--bulk", action="store_true", help="Process bulk images (default: False).")
    args = parser.parse_args()
    if args.bulk:
        for batter_pack in os.listdir(args.dataset_folder):
            for folder in os.listdir(os.path.join(args.dataset_folder, batter_pack)):
                image_path, ply_path, json_path, output_json_path, output_2d_det_path, output_image_path = get_file_paths(os.path.join(args.dataset_folder, batter_pack, folder), os.path.join(args.output_folder, batter_pack))
                process_screw_detection(image_path, ply_path, json_path, output_json_path, output_2d_det_path, output_image_path)
    else:
        image_path, ply_path, json_path, output_json_path, output_2d_det_path, output_image_path = get_file_paths(args.dataset_folder, args.output_folder)
        if args.visualize:
            process_screw_detection(image_path, ply_path, json_path, output_json_path, output_2d_det_path)
        else:
            process_screw_detection(image_path, ply_path, json_path, output_json_path, output_2d_det_path, output_image_path)
