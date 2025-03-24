"""
find the matched two images by PSNR and SSIM

python find_match.py \
    --pred_folder /home/hezongqi/ViewGS/work1/ViewCrafter/output_test/fortress/test0 \
    --gt_folder /home/hezongqi/Dataset/nerf_llff_data/fortress/images_8 \
    --output_dir ./match_results

"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from natsort import natsorted
import concurrent.futures
from functools import partial

# 使用scikit-image库计算PSNR和SSIM
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def find_image_files(folder):
    """查找文件夹中的所有图像文件"""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    files = []
    
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    
    return natsorted(files)

def process_image_pair(pred_img_path, gt_img_path, resize_to=None):
    """计算单对图像的PSNR和SSIM"""
    # 读取图像
    pred_img = cv2.imread(pred_img_path)
    gt_img = cv2.imread(gt_img_path)
    
    # 确保图像读取成功
    if pred_img is None or gt_img is None:
        print(f"Warning: Failed to load {pred_img_path} or {gt_img_path}")
        return None, None
    
    # 转换为RGB（OpenCV默认为BGR）
    pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    
    # 如果需要调整大小
    if resize_to is not None:
        # 获取原始尺寸比例
        h, w = pred_img.shape[:2]
        aspect_ratio = w / h
        
        # 计算新尺寸，保持纵横比
        if isinstance(resize_to, int):
            new_width = resize_to
            new_height = int(new_width / aspect_ratio)
        elif isinstance(resize_to, tuple) and len(resize_to) == 2:
            new_width, new_height = resize_to
        else:
            new_width, new_height = gt_img.shape[1], gt_img.shape[0]
        
        # 调整GT图像大小以匹配预测图像
        gt_img = cv2.resize(gt_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        pred_img = cv2.resize(pred_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        # 调整GT图像大小以匹配预测图像
        gt_img = cv2.resize(gt_img, (pred_img.shape[1], pred_img.shape[0]), interpolation=cv2.INTER_AREA)
    
    # 计算PSNR
    psnr_value = psnr(pred_img, gt_img)
    
    # 计算SSIM (多通道)
    ssim_value = ssim(pred_img, gt_img, channel_axis=2, data_range=255)
    
    return psnr_value, ssim_value

def process_one_pred(pred_img_path, gt_folder, gt_files, resize_to=None):
    """处理一个预测图像对应所有GT图像的匹配"""
    results = []
    pred_name = os.path.basename(pred_img_path)
    
    for gt_path in gt_files:
        psnr_value, ssim_value = process_image_pair(pred_img_path, gt_path, resize_to)
        if psnr_value is not None:
            results.append({
                'pred_path': pred_img_path,
                'pred_name': pred_name,
                'gt_path': gt_path,
                'gt_name': os.path.basename(gt_path),
                'psnr': psnr_value,
                'ssim': ssim_value
            })
    
    # 如果没有有效结果，返回None
    if not results:
        return None
    
    # 按PSNR排序结果
    results_by_psnr = sorted(results, key=lambda x: x['psnr'], reverse=True)
    best_psnr_match = results_by_psnr[0]
    
    # 按SSIM排序结果
    results_by_ssim = sorted(results, key=lambda x: x['ssim'], reverse=True)
    best_ssim_match = results_by_ssim[0]
    
    return {
        'pred_path': pred_img_path,
        'pred_name': pred_name,
        'best_psnr_match': best_psnr_match,
        'best_ssim_match': best_ssim_match,
        'all_results': results
    }

def find_best_matches(pred_folder, gt_folder, resize_to=None, max_workers=4):
    """查找最佳匹配的GT图像"""
    # 获取所有图像文件
    pred_files = find_image_files(pred_folder)
    gt_files = find_image_files(gt_folder)
    
    if not pred_files:
        raise ValueError(f"No image files found in {pred_folder}")
    
    if not gt_files:
        raise ValueError(f"No image files found in {gt_folder}")
    
    print(f"Found {len(pred_files)} prediction images and {len(gt_files)} ground truth images")
    
    # 使用线程池加速处理
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        process_func = partial(process_one_pred, gt_folder=gt_folder, gt_files=gt_files, resize_to=resize_to)
        futures = [executor.submit(process_func, pred_path) for pred_path in pred_files]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(pred_files), desc="Processing images"):
            result = future.result()
            if result:
                results.append(result)
    
    return results

def visualize_matches(results, output_dir=None):
    """可视化匹配结果"""
    if not results:
        print("No valid results to visualize")
        return
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, result in enumerate(results):
        pred_path = result['pred_path']
        best_psnr_match = result['best_psnr_match']
        best_ssim_match = result['best_ssim_match']
        
        # 读取图像
        pred_img = cv2.imread(pred_path)
        best_psnr_img = cv2.imread(best_psnr_match['gt_path'])
        best_ssim_img = cv2.imread(best_ssim_match['gt_path'])
        
        # 转换为RGB
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
        best_psnr_img = cv2.cvtColor(best_psnr_img, cv2.COLOR_BGR2RGB)
        best_ssim_img = cv2.cvtColor(best_ssim_img, cv2.COLOR_BGR2RGB)
        
        # 调整大小使所有图像具有相同尺寸
        height, width = pred_img.shape[:2]
        best_psnr_img = cv2.resize(best_psnr_img, (width, height))
        best_ssim_img = cv2.resize(best_ssim_img, (width, height))
        
        # 创建图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 显示图像
        axes[0].imshow(pred_img)
        axes[0].set_title(f"Prediction: {os.path.basename(pred_path)}")
        axes[0].axis('off')
        
        axes[1].imshow(best_psnr_img)
        axes[1].set_title(f"Best PSNR: {best_psnr_match['gt_name']}\nPSNR: {best_psnr_match['psnr']:.2f}")
        axes[1].axis('off')
        
        axes[2].imshow(best_ssim_img)
        axes[2].set_title(f"Best SSIM: {best_ssim_match['gt_name']}\nSSIM: {best_ssim_match['ssim']:.4f}")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"match_{idx}_{os.path.splitext(os.path.basename(pred_path))[0]}.png"))
            plt.close()
        else:
            plt.show()
    
    # 创建汇总报告
    if output_dir:
        with open(os.path.join(output_dir, "matching_report.csv"), 'w') as f:
            f.write("预测图片,最佳PSNR匹配,PSNR值,最佳SSIM匹配,SSIM值\n")
            for result in results:
                pred_name = result['pred_name']
                best_psnr_name = result['best_psnr_match']['gt_name']
                best_psnr_value = result['best_psnr_match']['psnr']
                best_ssim_name = result['best_ssim_match']['gt_name']
                best_ssim_value = result['best_ssim_match']['ssim']
                f.write(f"{pred_name},{best_psnr_name},{best_psnr_value:.4f},{best_ssim_name},{best_ssim_value:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='寻找最佳匹配的GT图片')
    parser.add_argument('--pred_folder', type=str, required=True, help='预测图片文件夹路径')
    parser.add_argument('--gt_folder', type=str, required=True, help='GT图片文件夹路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出可视化结果的文件夹')
    parser.add_argument('--resize', type=int, default=None, help='调整图像大小，保持纵横比')
    parser.add_argument('--workers', type=int, default=4, help='并行处理的线程数')
    
    args = parser.parse_args()
    
    # 查找最佳匹配
    results = find_best_matches(args.pred_folder, args.gt_folder, args.resize, args.workers)
    
    # 可视化结果
    visualize_matches(results, args.output_dir)
    
    # 输出总结
    print("\n最佳匹配结果:")
    print(f"{'预测图片':<30} | {'最佳PSNR匹配':<30} | {'PSNR值':<10} | {'最佳SSIM匹配':<30} | {'SSIM值':<10}")
    print("-" * 120)
    
    for result in results:
        pred_name = result['pred_name']
        best_psnr_name = result['best_psnr_match']['gt_name']
        best_psnr_value = result['best_psnr_match']['psnr']
        best_ssim_name = result['best_ssim_match']['gt_name']
        best_ssim_value = result['best_ssim_match']['ssim']
        
        print(f"{pred_name:<30} | {best_psnr_name:<30} | {best_psnr_value:<10.4f} | {best_ssim_name:<30} | {best_ssim_value:<10.4f}")

if __name__ == "__main__":
    main()