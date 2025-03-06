from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import json
import shutil
import threading
import logging
from main import process_screw_detection, visualize_point_cloud_and_screws_open3d

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


async def visualize_from_files(ply_path_str, output_json_path):
    # Load screws_frame from file
    with open(str(output_json_path), 'r') as f:
        screws_frame = json.load(f)["screws"]
    # Use a thread to run the visualization
    thread = threading.Thread(target=visualize_point_cloud_and_screws_open3d, args=(ply_path_str, screws_frame))
    thread.start()
    thread.join()


@app.post("/detect")
async def screw_detection(
    image: UploadFile = File(...),
    ply: UploadFile = File(...),
    transform: UploadFile = File(...),
    show_visualization: bool = False
):
    """API endpoint to process screw detection and return JSON and image results."""
    image_path = Path(UPLOAD_DIR) / image.filename
    ply_path = Path(UPLOAD_DIR) / ply.filename
    json_path = Path(UPLOAD_DIR) / transform.filename
    output_json_path = Path(RESULTS_DIR) / f"{image.filename}_screw_positions.json"
    output_image_path = Path(RESULTS_DIR) / f"{image.filename}_visualization.png"
    output_2d_det_path = Path(RESULTS_DIR) / f"{image.filename}_2d_detections.png"
    
    # Save uploaded files
    for file, path in zip([image, ply, transform], [image_path, ply_path, json_path]):
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    try:
        # Run processing
        img_bytes = process_screw_detection(image_path, ply_path, json_path, output_json_path, output_2d_det_path, output_image_path)
        if show_visualization:
            await visualize_from_files(ply_path, output_json_path)

    except Exception as e:
        logging.error(f"Error processing: {e}", exc_info=True)
        return None
    
    finally:
        for path in [image_path, ply_path, json_path]:
            if os.path.exists(path):
                os.remove(path)

    # print({"json_result": str(output_json_path), 
    #        "image_result": str(output_image_path),
    #        "2d detection_results": str(output_2d_det_path)})  

    logging.info(f'{{"json_result": "{output_json_path}", "image_result": "{output_image_path}", "2d detection_results": "{output_2d_det_path}"}}')

    return StreamingResponse(img_bytes, media_type="image/jpeg")


@app.post("/visualization")
async def detection_visualization(
    ply: UploadFile = File(...),
    detections: UploadFile = File(...),
):
    ply_path = Path(UPLOAD_DIR) / ply.filename
    detection_path = Path(UPLOAD_DIR) / detections.filename

    # Convert Path objects to strings before passing to the function
    ply_path_str = str(ply_path)
    detection_path_str = str(detection_path)
    
    # Save uploaded files
    for file, path in zip([ply, detections], [ply_path, detection_path]):
        with path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    
    try:
        await visualize_from_files(ply_path_str, detection_path_str)
    except Exception as e:
        print(e)
        return {"message" : "Visualization failed"}
    finally:
        for path in [ply_path, detection_path]:
            if os.path.exists(path):
                os.remove(path)

    return {"message": "Visualization started"}