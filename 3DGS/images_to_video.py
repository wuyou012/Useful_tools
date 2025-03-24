#!/usr/bin/env python3
# filepath: /home/hezongqi/images_to_video.py

import os
import argparse
import glob
from tqdm import tqdm
import cv2
import numpy as np
from natsort import natsorted

def create_video_from_images(input_folder, output_file, fps=30, resolution=None, quality=95, verbose=True):
    """
    将文件夹中的图片合成为MP4视频（不包括子目录中的图片）
    
    Args:
        input_folder (str): 包含图片的文件夹路径
        output_file (str): 输出视频文件路径
        fps (int): 视频帧率，默认30
        resolution (tuple): 输出视频分辨率 (width, height)，默认使用第一张图片的分辨率
        quality (int): 输出视频质量 (0-100)，默认95
        verbose (bool): 是否显示详细信息，默认True
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    # 查找所有图片文件（只在当前目录中查找，不包括子目录）
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    
    for ext in extensions:
        # 仅匹配直接位于input_folder下的文件
        for img_file in glob.glob(os.path.join(input_folder, ext)):
            # 确认不是子目录
            if os.path.isfile(img_file):
                image_files.append(img_file)
        
        # 同样处理大写扩展名
        for img_file in glob.glob(os.path.join(input_folder, ext.upper())):
            if os.path.isfile(img_file):
                image_files.append(img_file)
    
    # 确保有图片文件
    if not image_files:
        print(f"错误：在 {input_folder} 中没有找到图片文件")
        return False
    
    # 按文件名自然排序
    image_files = natsorted(image_files)
    
    if verbose:
        print(f"找到 {len(image_files)} 张图片")
        print(f"第一张图片: {os.path.basename(image_files[0])}")
        print(f"最后一张图片: {os.path.basename(image_files[-1])}")
    
    # 读取第一张图片获取尺寸
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"错误：无法读取图片 {image_files[0]}")
        return False
    
    height, width = first_image.shape[:2]
    
    # 使用指定分辨率
    if resolution:
        width, height = resolution
    
    # 创建输出文件夹
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 设置编码器和参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'avc1' 编码器
    video_writer = cv2.VideoWriter(
        output_file, 
        fourcc, 
        fps, 
        (width, height)
    )
    
    # 设置视频质量
    video_writer.set(cv2.VIDEOWRITER_PROP_QUALITY, quality)
    
    if verbose:
        print(f"正在创建视频: {output_file}")
        print(f"分辨率: {width}x{height}, FPS: {fps}, 质量: {quality}")
    
    # 处理每一张图片
    for image_file in tqdm(image_files, desc="处理图片", disable=not verbose):
        img = cv2.imread(image_file)
        
        if img is None:
            print(f"警告：无法读取图片 {image_file}，跳过")
            continue
        
        # 调整尺寸以匹配输出视频分辨率
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # 写入视频
        video_writer.write(img)
    
    # 释放资源
    video_writer.release()
    
    if verbose:
        print(f"视频已保存到: {output_file}")
        print(f"总帧数: {len(image_files)}, 时长: {len(image_files)/fps:.2f}秒")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='将文件夹中的图片合成为MP4视频（不包括子目录）')
    parser.add_argument('--input', '-i', type=str, required=True, help='包含图片的文件夹路径')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出视频文件路径')
    parser.add_argument('--fps', type=int, default=30, help='视频帧率，默认30')
    parser.add_argument('--width', type=int, default=None, help='输出视频宽度，默认使用图片原始宽度')
    parser.add_argument('--height', type=int, default=None, help='输出视频高度，默认使用图片原始高度')
    parser.add_argument('--quality', type=int, default=95, help='输出视频质量(0-100)，默认95')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    # 设置分辨率
    resolution = None
    if args.width and args.height:
        resolution = (args.width, args.height)
    
    # 创建视频
    create_video_from_images(
        args.input, 
        args.output, 
        fps=args.fps, 
        resolution=resolution, 
        quality=args.quality,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()