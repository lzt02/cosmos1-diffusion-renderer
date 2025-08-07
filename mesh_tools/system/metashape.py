# system/Metashape.py
import Metashape
import os
import glob
from utils.dataload import load_colmap_data, load_json_data
import numpy as np
import json
from pathlib import Path


class TextureProcessor:
    def __init__(self, enable_gpu=True):
        """初始化并配置 GPU"""
        if enable_gpu:
            Metashape.app.gpu_mask = 2**len(Metashape.app.enumGPUDevices()) - 1
            Metashape.app.cpu_enable = False

    def import_colmap_camera_params(self, chunk, cameras):
        """Import camera intrinsics"""
        print("Importing camera intrinsics from COLMAP...")
        if cameras is None:
            print("No camera file provided, using default intrinsics.")
            for camera in chunk.cameras:
                width = 1024
                height = 1024
                sensor_width_mm = 36.0
                default_focal_px = 0.9 * width
                pixel_size_mm = sensor_width_mm / width

                camera.sensor.type = Metashape.Sensor.Type.Frame
                camera.sensor.width = width
                camera.sensor.height = height
                camera.sensor.focal_length = default_focal_px * pixel_size_mm
                camera.sensor.pixel_size = (pixel_size_mm, pixel_size_mm)
                camera.sensor.calibration.cx = 0.5
                camera.sensor.calibration.cy = 0.5
            return
        # 应用相机内参到 chunk 中
        for camera_id, cam in cameras.items():
            model = cam['model']
            width = cam['width']
            height = cam['height']
            params = cam['params']

            for camera in chunk.cameras:
                if camera.sensor.width != width or camera.sensor.height != height:
                    camera.sensor.width = width
                    camera.sensor.height = height

                if model == "PINHOLE":
                    fx, fy, cx, cy = params
                    sensor_width_mm = 36.0
                    pixel_size_mm = sensor_width_mm / width
                    camera.sensor.focal_length = fx * pixel_size_mm
                    camera.sensor.pixel_size = (pixel_size_mm, pixel_size_mm)
                    camera.sensor.calibration.cx = cx / width
                    camera.sensor.calibration.cy = 1 - cy / height

                elif model == "SIMPLE_RADIAL":
                    f, cx, cy, k1 = params
                    sensor_width_mm = 36.0
                    pixel_size_mm = sensor_width_mm / width
                    camera.sensor.focal_length = f * pixel_size_mm
                    camera.sensor.pixel_size = (pixel_size_mm, pixel_size_mm)
                    camera.sensor.calibration.cx = cx / width
                    camera.sensor.calibration.cy = 1 - cy / height
                        
    def import_colmap_cameras(self, chunk, images, w2c=True):
        """Import camera extrinsics"""
        print("Importing camera poses from COLMAP...")
        
        sf = 1.0
        for camera in chunk.cameras:
            for image in images:
                name_no_ext = image['name'].rsplit('.', 1)[0]
                if camera.label == name_no_ext:
                    qvec = np.array([image['qw'], image['qx'], image['qy'], image['qz']])
                    tvec = np.array([[image['tx'] * sf, image['ty'] * sf, image['tz'] * sf]])

                    w, x, y, z = qvec

                    R = np.array([
                        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
                        [2*x*y + 2*w*z,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
                        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x**2 - 2*y**2]
                    ])
                    
                    if w2c: # need convert to c2w
                        R = R.T
                        t = (-R @ tvec.T)
                    else:
                        R[:, 2] *= -1
                        R[:, 1] *= -1
                        t = tvec.T

                    cam_matrix = Metashape.Matrix([
                        [R[0, 0], R[0, 1], R[0, 2], t[0, 0]],
                        [R[1, 0], R[1, 1], R[1, 2], t[1, 0]],
                        [R[2, 0], R[2, 1], R[2, 2], t[2, 0]],
                        [0, 0, 0, 1]
                    ])
                    camera.transform = cam_matrix
                    camera.reference.location = camera.transform.translation()
                    break
            
    def texture_mesh(self, images_folder, cameras_folder, obj_file, output_file, texture_size=4096, format="colmap"):

        doc = Metashape.Document()
        chunk = doc.addChunk()

        img_exts = ('.jpg', '.png', '.jpeg')
        images = [
            os.path.join(images_folder, f) 
            for f in os.listdir(images_folder) 
            if f.lower().endswith(img_exts)
        ]
        chunk.addPhotos(images)

        if os.path.isfile(cameras_folder) and cameras_folder.endswith(".json"):
            camera_file = None
            image_file = cameras_folder
            cameras = load_json_data(image_file)
            camera_params = None
            self.import_colmap_camera_params(chunk, camera_params)
            self.import_colmap_cameras(chunk, cameras, w2c=False)
        else:
            camera_files = glob.glob(os.path.join(cameras_folder, "cameras*"))
            image_files = glob.glob(os.path.join(cameras_folder, "images*"))
            camera_file = camera_files[0]
            image_file = image_files[0]
            camera_params , cameras = load_colmap_data(camera_file, image_file)
        
            self.import_colmap_camera_params(chunk, camera_params)
            self.import_colmap_cameras(chunk, cameras)

        chunk.importModel(obj_file, format=Metashape.ModelFormatOBJ)
        chunk.buildTexture(
            blending_mode=Metashape.MosaicBlending,
            texture_size=texture_size,
            fill_holes=True,
            ghosting_filter=True
        )
        
        chunk.exportModel(output_file, format=Metashape.ModelFormatOBJ, save_texture=True)
    
    
    def render_camera_views(self, cameras_folder, obj_file):
        doc = Metashape.Document()
        chunk = doc.addChunk()

        cameras_path = None
        images_path = os.path.join(cameras_folder, "camera_pose.json")
        with open(images_path, 'r') as f:
            data = json.load(f)

        poses = data["poses_by_sample"]

        for key, pose in poses.items():
            file_path = pose["file_path"]  
            label = Path(file_path).stem  
            camera = chunk.addCamera()
            camera.label = label
            camera.sensor = chunk.addSensor()
            camera.sensor.type = Metashape.Sensor.Type.Frame  
            print(f"Created camera: {label}")
        cameras = load_json_data(images_path)    
        camera_params = None
        self.import_colmap_camera_params(chunk, camera_params)
        self.import_colmap_cameras(chunk, cameras, w2c=False)
        
        print("Importing mesh...")
        chunk.importModel(obj_file, format=Metashape.ModelFormatOBJ)
        output_folder = os.path.join(os.path.dirname(obj_file), "images")
        os.makedirs(output_folder, exist_ok=True)
        
        for camera in chunk.cameras:
            if not camera.transform:
                continue
                
            output_file = os.path.join(output_folder, f"{camera.label}.png")
            image = chunk.model.renderImage(
                transform=camera.transform,
                calibration=camera.sensor.calibration
            )
            image.save(output_file)