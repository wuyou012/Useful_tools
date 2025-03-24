"""
python convert_bin2txt.py --scene flower --type cameras

change root and suffix for custom
"""



import os
import numpy as np
import struct
import argparse
from collections import namedtuple

# 定义数据结构
CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
# 定义相机模型常量
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
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                        for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                          for camera_model in CAMERA_MODELS])

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

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_binary(path_to_model_file):
    """读取COLMAP的points3D.bin文件"""
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length, format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_extrinsics_binary(path_to_model_file):
    """读取COLMAP的images.bin文件"""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
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

def convert_points3D_bin_to_txt(bin_path):
    """将COLMAP的points3D.bin文件转换为points3D.txt格式"""
    # 确定输出路径
    txt_path = os.path.splitext(bin_path)[0] + '.txt'
    
    # 打开txt文件准备写入
    with open(txt_path, 'w') as txt_file:
        # 写入文件头
        txt_file.write("# 3D point list with one point per line:\n")
        txt_file.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        # 打开二进制文件来获取点信息
        with open(bin_path, "rb") as bin_file:
            # 读取点的数量
            num_points = read_next_bytes(bin_file, 8, "Q")[0]
            
            # 遍历每个点
            for p_id in range(num_points):
                # 读取点的属性
                binary_point_line_properties = read_next_bytes(
                    bin_file, num_bytes=43, format_char_sequence="QdddBBBd")
                
                # 解析点的ID、坐标、颜色和误差
                point_id = binary_point_line_properties[0]
                x, y, z = binary_point_line_properties[1:4]
                r, g, b = binary_point_line_properties[4:7]
                error = binary_point_line_properties[7]
                
                # 读取track信息
                track_length = read_next_bytes(
                    bin_file, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = read_next_bytes(
                    bin_file, num_bytes=8*track_length,
                    format_char_sequence="ii"*track_length)
                
                # 写入点的基本信息
                txt_file.write(f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error}")
                
                # 写入track信息
                for i in range(track_length):
                    image_id = track_elems[2*i]
                    point2D_idx = track_elems[2*i+1]
                    txt_file.write(f" {image_id} {point2D_idx}")
                
                txt_file.write("\n")
    
    print(f"已成功将 {bin_path} 转换为 {txt_path}")
    return txt_path

def convert_images_bin_to_txt(bin_path):
    """将COLMAP的images.bin文件转换为images.txt格式"""
    # 确定输出路径
    txt_path = os.path.splitext(bin_path)[0] + '.txt'
    
    # 打开txt文件准备写入
    with open(txt_path, 'w') as txt_file:
        # 写入文件头
        txt_file.write("# Image list with two lines of data per image:\n")
        txt_file.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        txt_file.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        # 读取二进制文件
        with open(bin_path, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            
            for _ in range(num_reg_images):
                binary_image_properties = read_next_bytes(
                    fid, num_bytes=64, format_char_sequence="idddddddi")
                image_id = binary_image_properties[0]
                qvec = binary_image_properties[1:5]
                tvec = binary_image_properties[5:8]
                camera_id = binary_image_properties[8]
                
                # 读取图像名称
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":   # 查找ASCII 0结束符
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                
                # 写入图像信息的第一行
                txt_file.write(f"{image_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {camera_id} {image_name}\n")
                
                # 读取2D点信息
                num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
                x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                         format_char_sequence="ddq"*num_points2D)
                
                # 写入图像信息的第二行 (特征点信息)
                point2d_line = " ".join([f"{x_y_id_s[i*3]} {x_y_id_s[i*3+1]} {x_y_id_s[i*3+2]}" 
                                      for i in range(num_points2D)])
                txt_file.write(f"{point2d_line}\n")
                
    print(f"已成功将 {bin_path} 转换为 {txt_path}")
    return txt_path

def convert_cameras_bin_to_txt(bin_path):
    """将COLMAP的cameras.bin文件转换为cameras.txt格式"""
    # 确定输出路径
    txt_path = os.path.splitext(bin_path)[0] + '.txt'
    
    # 打开txt文件准备写入
    with open(txt_path, 'w') as txt_file:
        # 写入文件头
        txt_file.write("# Camera list with one line of data per camera:\n")
        txt_file.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        # 读取二进制文件
        with open(bin_path, "rb") as bin_file:
            # 读取相机数量
            num_cameras = read_next_bytes(bin_file, 8, "Q")[0]
            
            # 遍历每个相机
            for _ in range(num_cameras):
                # 读取相机属性
                camera_properties = read_next_bytes(
                    bin_file, num_bytes=24, format_char_sequence="iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                width = camera_properties[2]
                height = camera_properties[3]
                
                # 获取相机模型信息
                model_name = CAMERA_MODEL_IDS[model_id].model_name
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                
                # 读取相机参数
                params = read_next_bytes(bin_file, num_bytes=8*num_params,
                                        format_char_sequence="d"*num_params)
                
                # 构建参数字符串
                params_str = " ".join([f"{param}" for param in params])
                
                # 写入相机信息
                txt_file.write(f"{camera_id} {model_name} {width} {height} {params_str}\n")
    
    print(f"已成功将 {bin_path} 转换为 {txt_path}")
    return txt_path

def main():
    parser = argparse.ArgumentParser(description="COLMAP二进制文件与文本文件相互转换工具")
    parser.add_argument("--scene", type=str, required=True, help="二进制文件路径")
    parser.add_argument("--type", type=str, choices=["cameras", "images", "points3D"], required=True, help="文件类型")
    
    args = parser.parse_args()
    root = '/home/hezongqi/Dataset/'
    suffix = 'sparse/0/'
    bin_path = os.path.join(root, args.scene, suffix, f'{args.type}.bin')
    
    if args.type == "cameras":
        txt_file_path = convert_cameras_bin_to_txt(bin_path)
    elif args.type == "images":
        txt_file_path = convert_images_bin_to_txt(bin_path)
    else:  # points3D
        txt_file_path = convert_points3D_bin_to_txt(bin_path)
    
    print(f"文件已成功转换并保存至: {txt_file_path}")

if __name__ == "__main__":
    main()