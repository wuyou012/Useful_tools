"""
average score from json file
{
    "PSNR": [
        17.6706
    ],
    "SSIM": [
        0.6324
    ],
    "LPIPS": [
        0.2522
    ],
    "scene": "Ballroom"
}

python average_json.py json_path

"""
import json
import os
import sys
import re
import numpy as np

def clean_json_file(file_path):
    """修复不正确的JSON格式"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 将多个JSON对象拼接成一个有效的JSON数组
    content = '[' + re.sub(r'\}\s*\{', '},{', content) + ']'
    
    return json.loads(content)

def calculate_metrics_average(file_path):
    """计算JSON文件中的平均PSNR、SSIM和LPIPS值"""
    try:
        # 尝试加载并修复JSON
        data = clean_json_file(file_path)
        
        # 提取所有指标
        psnr_values = []
        ssim_values = []
        lpips_values = []
        
        for scene_data in data:
            psnr_values.extend(scene_data.get('PSNR', []))
            ssim_values.extend(scene_data.get('SSIM', []))
            lpips_values.extend(scene_data.get('LPIPS', []))
            
        # 计算平均值
        avg_psnr = np.mean(psnr_values) if psnr_values else 0
        avg_ssim = np.mean(ssim_values) if ssim_values else 0
        avg_lpips = np.mean(lpips_values) if lpips_values else 0
        
        return {
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim,
            'avg_lpips': avg_lpips,
            'num_scenes': len(data),
            'scene_details': [{
                'scene': scene.get('scene', f'Scene {i+1}'),
                'psnr': scene.get('PSNR', [0])[0] if scene.get('PSNR') else 0,
                'ssim': scene.get('SSIM', [0])[0] if scene.get('SSIM') else 0,
                'lpips': scene.get('LPIPS', [0])[0] if scene.get('LPIPS') else 0
            } for i, scene in enumerate(data)]
        }
    except Exception as e:
        print(f"错误: {e}")
        return None

def main():
    # 默认文件路径
    default_file_path = "/home/hezongqi/ViewGS/work1/work5/output_diff0/test_results.json"
    
    # 允许从命令行指定文件路径
    file_path = sys.argv[1] if len(sys.argv) > 1 else default_file_path
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return
    
    # 计算平均值
    results = calculate_metrics_average(file_path)
    
    if results:
        print("\n============ 平均指标 ============")
        print(f"平均 PSNR:  {results['avg_psnr']:.4f} dB")
        print(f"平均 SSIM:  {results['avg_ssim']:.4f}")
        print(f"平均 LPIPS: {results['avg_lpips']:.4f}")
        print(f"场景数量:   {results['num_scenes']}")
        
        print("\n============ 场景详情 ============")
        scene_data = results['scene_details']
        
        # 创建一个表格
        header = f"{'场景名称':<15} {'PSNR (dB)':<12} {'SSIM':<12} {'LPIPS':<12}"
        print(header)
        print("-" * len(header))
        
        for scene in scene_data:
            print(f"{scene['scene']:<15} {scene['psnr']:<12.4f} {scene['ssim']:<12.4f} {scene['lpips']:<12.4f}")
            
        # 生成报告文件
        report_path = os.path.join(os.path.dirname(file_path), "metrics_summary.txt")
        with open(report_path, 'w') as f:
            f.write("============ 平均指标 ============\n")
            f.write(f"平均 PSNR:  {results['avg_psnr']:.4f} dB\n")
            f.write(f"平均 SSIM:  {results['avg_ssim']:.4f}\n")
            f.write(f"平均 LPIPS: {results['avg_lpips']:.4f}\n")
            f.write(f"场景数量:   {results['num_scenes']}\n\n")
            
            f.write("============ 场景详情 ============\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            
            for scene in scene_data:
                f.write(f"{scene['scene']:<15} {scene['psnr']:<12.4f} {scene['ssim']:<12.4f} {scene['lpips']:<12.4f}\n")
                
        print(f"\n报告已保存至: {report_path}")
        
if __name__ == "__main__":
    main()