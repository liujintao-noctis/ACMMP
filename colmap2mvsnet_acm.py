#!/usr/bin/env python
"""
Copyright 2019, Jingyang Zhang and Yao Yao, HKUST. Model reading is provided by COLMAP.
Preprocess script.
View selection is modified according to COLMAP's strategy, Qingshan Xu
"""

from __future__ import print_function
import collections
import struct
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import argparse
import shutil
import cv2

# ============================ read_model.py ============================#
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """Reads COLMAP cameras from a text file."""
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """Reads COLMAP cameras from a binary file."""
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """Reads COLMAP images from a text file."""
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path_to_model_file):
    """Reads COLMAP images from a binary file."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_points3D_text(path):
    """Reads COLMAP 3D points from a text file."""
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """Reads COLMAP 3D points from a binary file."""
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(cameras_path, images_path, points3d_path, ext):
    """Modified to read specific file paths."""
    if ext == ".txt":
        print(f"Reading model files (TEXT format)...")
        cameras = read_cameras_text(cameras_path)
        images = read_images_text(images_path)
        points3D = read_points3D_text(points3d_path)
    else:
        print(f"Reading model files (BINARY format)...")
        cameras = read_cameras_binary(cameras_path)
        images = read_images_binary(images_path)
        points3D = read_points3d_binary(points3d_path)
    
    # 增加打印信息
    print(f"Found {len(cameras)} cameras, {len(images)} registered images, and {len(points3D)} 3D points.")
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec



def calc_score(inputs, images, points3d, extrinsic, args):
    i, j = inputs
    id_i = images[i+1].point3D_ids
    id_j = images[j+1].point3D_ids
    # 使用 np.intersect1d 更高效地找到共同的 3D 点 ID
    id_intersect = np.intersect1d(id_i[id_i != -1], id_j[id_j != -1])
    
    cam_center_i = -np.matmul(extrinsic[i+1][:3, :3].transpose(), extrinsic[i+1][:3, 3:4])[:, 0]
    cam_center_j = -np.matmul(extrinsic[j+1][:3, :3].transpose(), extrinsic[j+1][:3, 3:4])[:, 0]
    score = 0
    angles = []
    
    # 增加对 id_intersect 长度的检查
    if len(id_intersect) == 0:
        return i, j, 0.0
        
    for pid in id_intersect:
        p = points3d[pid].xyz
        
        # 优化：避免重复计算范数和点乘
        vec_i = cam_center_i - p
        vec_j = cam_center_j - p
        norm_i = np.linalg.norm(vec_i)
        norm_j = np.linalg.norm(vec_j)
        
        # 避免除以零（如果点P恰好是相机中心）
        if norm_i == 0 or norm_j == 0:
            continue
            
        cosine_angle = np.dot(vec_i, vec_j) / (norm_i * norm_j)
        # 确保 cosine_angle 在 [-1, 1] 范围内，避免浮点误差导致 arccos 报错
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        theta = (180 / np.pi) * np.arccos(cosine_angle) # triangulation angle
        angles.append(theta)
        score += 1
        
    if len(angles) > 0:
        angles_sorted = sorted(angles)
        triangulationangle = angles_sorted[int(len(angles_sorted) * 0.75)]
        if triangulationangle < 1:
            score = 0.0
    return i, j, score

def processing_single_scene(args):
    print("\n" + "="*50)
    print("🎬 Starting MVSNet Preprocessing...")
    print(f"Source Image Dir: {args.image_dir}")
    print(f"Output Save Dir: {args.save_folder}")
    print("="*50 + "\n")

    # --- 路径变量设置和目录准备 ---
    image_dir = args.image_dir
    cam_dir = os.path.join(args.save_folder, 'cams')
    image_converted_dir = os.path.join(args.save_folder, 'images')

    print(f"1. Preparing output directories: {args.save_folder}")
    if os.path.exists(image_converted_dir):
        print("  - Removing existing image directory.")
        shutil.rmtree(image_converted_dir)
    os.makedirs(image_converted_dir)
    if os.path.exists(cam_dir):
        print("  - Removing existing cam directory.")
        shutil.rmtree(cam_dir)
    os.makedirs(cam_dir)
    print("  - Output directories ready.")
    
    # --- 模型文件读取 ---
    print("\n2. Reading COLMAP model files...")
    cameras, images, points3d = read_model(args.cameras_file, args.images_file, args.points3d_file, args.model_ext)
    num_images = len(list(images.items()))
    print(f"Total {num_images} images to process.")

    param_type = {
        'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
        'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
        'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
        'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
        'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
        'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
        'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
        'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
        'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
        'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
        'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
    }

    # intrinsic & extrinsic map creation
    intrinsic = {}
    extrinsic = {}
    new_images = {}
    
    for i, image_id in enumerate(sorted(images.keys())):
        image = images[image_id]
        
        # 重新映射 image ID (从 1 开始)
        new_images[i+1] = image
        
        # 内参 (Intrinsic)
        cam = cameras[image.camera_id]
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if 'f' in param_type[cam.model]:
            params_dict['fx'] = params_dict['f']
            params_dict['fy'] = params_dict['f']
        i_mat = np.array([
            [params_dict['fx'], 0, params_dict['cx']],
            [0, params_dict['fy'], params_dict['cy']],
            [0, 0, 1]
        ])
        intrinsic[image.camera_id] = i_mat
        
        # 外参 (Extrinsic)
        e = np.zeros((4, 4))
        e[:3, :3] = qvec2rotmat(image.qvec)
        e[:3, 3] = image.tvec
        e[3, 3] = 1
        extrinsic[i+1] = e # 注意：这里使用新的 (i+1) ID
        
    images = new_images
    print(f"  - Calculated {len(intrinsic)} unique intrinsic matrices.")
    print(f"  - First Intrinsic Matrix (Camera ID {list(intrinsic.keys())[0]}):\n{list(intrinsic.values())[0]}")
    print(f"  - First Extrinsic Matrix (Image ID 1):\n{extrinsic[1]}")

    # --- 深度范围和间隔计算 ---
    print("\n3. Calculating depth ranges...")
    depth_ranges = {}
    for i in range(num_images):
        zs = []
        for p3d_id in images[i+1].point3D_ids:
            if p3d_id == -1:
                continue
            transformed = np.matmul(extrinsic[i+1], [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1], points3d[p3d_id].xyz[2], 1])
            # 修复：使用 .item() 替代 np.asscalar()
            zs.append(transformed[2].item()) 
        
        # 确保 zs 不为空
        if not zs:
            print(f"  - Warning: Image {i} has no valid 3D points. Skipping depth range calculation for this image.")
            depth_ranges[i+1] = (0.0, 0.0, args.max_d, 0.0) # 使用默认值或一个安全值
            continue

        zs_sorted = sorted(zs)
        # relaxed depth range
        depth_min = zs_sorted[int(len(zs) * .01)] * 0.75
        depth_max = zs_sorted[int(len(zs) * .99)] * 1.25
        
        # determine depth number (simplified calculation)
        if args.max_d == 0:
            # 复杂的逆深度计算 (保留原逻辑)
            image_int = intrinsic[images[i+1].camera_id]
            image_ext = extrinsic[i+1]
            image_r = image_ext[0:3, 0:3]
            image_t = image_ext[0:3, 3]
            p1 = [image_int[0, 2], image_int[1, 2], 1]
            p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
            P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
            P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
            P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
            P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
            depth_num = (1 / depth_min - 1 / depth_max) / (1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
        else:
            depth_num = args.max_d
            
        depth_num = int(round(depth_num)) # 确保深度数量是整数
        depth_interval = (depth_max - depth_min) / (depth_num - 1) / args.interval_scale
        depth_ranges[i+1] = (depth_min, depth_interval, depth_num, depth_max)
        
    print(f"  - First Depth Range (Image 1): min={depth_ranges[1][0]:.3f}, num={depth_ranges[1][2]}, interval={depth_ranges[1][1]:.6f}")

    # --- 视图选择 ---
    print("\n4. Performing view selection (Multi-processing)...")
    score = np.zeros((len(images), len(images)))
    queue = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            queue.append((i, j))

    # 使用 with 语句管理进程池，确保资源正确释放
    with mp.Pool(processes=mp.cpu_count()) as p:
        func = partial(calc_score, images=images, points3d=points3d, args=args, extrinsic=extrinsic)
        result = p.map(func, queue)
        
    for i, j, s in result:
        score[i, j] = s
        score[j, i] = s
    
    view_sel = []
    num_view = min(20, len(images) - 1)
    for i in range(len(images)):
        sorted_score = np.argsort(score[i])[::-1]
        # 过滤掉得分为 0 的视图 (可选，但常用)
        valid_scores = [(k, score[i, k]) for k in sorted_score if score[i, k] > 0]
        view_sel.append(valid_scores[:num_view])
    print(f"  - View Selection Complete. Max {num_view} neighbors selected per image.")
    print(f"  - First Image neighbors (ID 0): {view_sel[0]}")

    # --- 写入 CAM 文件和 PAIR 文件 ---
    print("\n5. Writing CAM files and pair.txt...")
    
    # 写入 CAM 文件
    for i in range(num_images):
        # 增加对 depth_ranges 缺失键的检查
        if i + 1 not in depth_ranges:
            print(f"  - Skipping CAM file for Image {i} (no valid depth range).")
            continue
            
        with open(os.path.join(cam_dir, '%08d_cam.txt' % i), 'w') as f:
            f.write('extrinsic\n')
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i+1][j, k]) + ' ')
                f.write('\n')
            f.write('\nintrinsic\n')
            
            # 增加对 intrinsic 缺失键的检查 (不太可能，但更安全)
            cam_id = images[i+1].camera_id
            if cam_id not in intrinsic:
                 print(f"  - Error: Camera ID {cam_id} not found in intrinsic map.")
                 continue

            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[cam_id][j, k]) + ' ')
                f.write('\n')
                
            f.write('\n%f %f %f %f\n' % (depth_ranges[i+1][0], depth_ranges[i+1][1], depth_ranges[i+1][2], depth_ranges[i+1][3]))

    # 写入 pair.txt
    with open(os.path.join(args.save_folder, 'pair.txt'), 'w') as f:
        f.write('%d\n' % len(images))
        for i, sorted_score in enumerate(view_sel):
            # i 是从 0 开始的索引
            f.write('%d\n%d ' % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                # image_id 是从 1 开始的 key，但 MVSNet pair.txt 通常需要从 0 开始的 index
                # 这里的 image_id 来自 sorted_score，是另一个图像的 0-based index k
                # 检查原逻辑: sorted_score 存储的是 (k, score)，k 是 0-based index
                f.write('%d %d ' % (image_id, int(round(s)))) # 将 score 转换为整数
            f.write('\n')
    print("  - CAM files and pair.txt successfully written.")


    # --- 图像转换 ---
    print("\n6. Converting and copying images to output folder...")
    success_count = 0
    for i in range(num_images):
        img_info = images[i + 1]
        img_path = os.path.join(image_dir, img_info.name)
        
        # 检查源文件是否存在
        if not os.path.exists(img_path):
            print(f"  - Error: Source image file not found: {img_path}. Skipping.")
            continue

        target_path = os.path.join(image_converted_dir, '%08d.jpg' % i)

        if not img_path.lower().endswith((".jpg", ".jpeg")):
            # 需要转换格式
            img = cv2.imread(img_path)
            if img is None:
                print(f"  - Warning: Could not read image {img_path} with OpenCV. Skipping.")
                continue
            cv2.imwrite(target_path, img)
        else:
            # 已经是 JPG，直接复制
            shutil.copyfile(img_path, target_path)
            
        success_count += 1
        
    print(f"  - Successfully processed {success_count}/{num_images} images.")
    print("\n" + "="*50)
    print("✅ Preprocessing Complete!")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert colmap camera')

    # --- 新增和修改的参数 ---
    parser.add_argument('--image_dir', required=True, type=str, help='Directory containing the source images.')
    parser.add_argument('--cameras_file', required=True, type=str, help='Path to the COLMAP cameras file (cameras.ext).')
    parser.add_argument('--images_file', required=True, type=str, help='Path to the COLMAP images file (images.ext).')
    parser.add_argument('--points3d_file', required=True, type=str, help='Path to the COLMAP 3D points file (points3D.ext).')
    parser.add_argument('--save_folder', required=True, type=str, help='Output folder to save MVSNet format data.')
    # --- END 新增和修改的参数 ---

    parser.add_argument('--max_d', type=int, default=192, help='Max number of depth layers. 0 for inverse depth calculation.')
    parser.add_argument('--interval_scale', type=float, default=1, help='Scale factor for depth interval.')

    parser.add_argument('--theta0', type=float, default=5, help='Triangulation angle score threshold.')
    parser.add_argument('--sigma1', type=float, default=1, help='Sigma for angle score < theta0.')
    parser.add_argument('--sigma2', type=float, default=10, help='Sigma for angle score >= theta0.')
    parser.add_argument('--model_ext', type=str, default=".txt",  choices=[".txt", ".bin"], help='sparse model extension (.txt or .bin).')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory not found: {args.image_dir}")
    if not os.path.exists(args.cameras_file):
        raise FileNotFoundError(f"Cameras file not found: {args.cameras_file}")
    if not os.path.exists(args.images_file):
        raise FileNotFoundError(f"Images file not found: {args.images_file}")
    if not os.path.exists(args.points3d_file):
        raise FileNotFoundError(f"3D Points file not found: {args.points3d_file}")
    
    os.makedirs(os.path.join(args.save_folder), exist_ok=True)
    processing_single_scene(args)