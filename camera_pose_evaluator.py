#!/usr/bin/env python3
# filepath: /home/hezongqi/camera_pose_evaluator_simple.py
"""
python camera_pose_evaluator.py \
    --source_a /home/hezongqi/ViewGS/work1/ViewCrafter/output_TT/Ballroom \
    --type_a dust \
    --source_b /home/data1/Tanks/Ballroom \
    --type_b colmap \
    --output_dir ./pose_comparison
"""
def attach_debugger():
    import debugpy
    debugpy.listen(5678)
    print("Waiting for Debuger to Attach on Port 5678...")
    debugpy.wait_for_client()
    print("Attached!")
# attach_debugger()

import os
import sys
import torch
import numpy as np
import argparse
from copy import copy
import matplotlib.pyplot as plt
from natsort import natsorted
from pathlib import Path

# 导入位姿评估和可视化所需的函数
sys.path.append('/home/hezongqi/Colmap-free/CF-3DGS')
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_ATE, compute_rpe
from utils.vis_utils import plot_pose

# 导入场景加载函数
sys.path.append('/home/hezongqi/ViewGS/work1/work5')
from scene.dataset_readers import readDustInfo, readColmapSceneInfo_noply, getNerfppNorm, readinstantSceneInfo


class CameraPoseEvaluator:
    """简化版相机位姿评估器：读取、对齐、比较和可视化不同来源的相机位姿"""
    
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载两组相机位姿
        self.poses_a = self.load_poses(args.source_a, args.type_a)
        self.poses_b = self.load_poses(args.source_b, args.type_b)
        
        # 确保两组位姿长度一致
        min_len = min(len(self.poses_a), len(self.poses_b))
        self.poses_a = self.poses_a[:min_len]
        self.poses_b = self.poses_b[:min_len]
        
        print(f"加载了 {min_len} 对相机位姿")
        
    def load_poses(self, source_path, data_type):
        """从指定来源和类型加载相机位姿"""
        poses = []
        
        if data_type.lower() == 'dust':
            # 使用 readDustInfo 加载相机位姿
            print(f"从 {source_path} 加载 Dust 相机位姿...")
            
            # 创建简化的 args 对象以符合 readDustInfo 接口
            class Args:
                def __init__(self):
                    self.sub_ratio = 20
                    
            scene_info, _ = readDustInfo(
                Args(),
                path=source_path, 
                images=None, 
                eval=True, 
                colmap_path='', 
                n_views=3
            )
            
            # 提取位姿
            for cam_info in scene_info.train_cameras:
                pose = np.eye(4)
                pose[:3, :3] = cam_info.R
                pose[:3, 3] = cam_info.T
                poses.append(torch.from_numpy(pose).float())
            # for cam_info in scene_info.test_cameras:
            #     pose = np.eye(4)
            #     pose[:3, :3] = cam_info.R
            #     pose[:3, 3] = cam_info.T
            #     poses.append(torch.from_numpy(pose).float())
                
        elif data_type.lower() == 'colmap':
            # 使用 readColmapSceneInfo 加载相机位姿
            print(f"从 {source_path} 加载 Colmap 相机位姿...")
            
            scene_info = readColmapSceneInfo_noply(
                path=source_path,
                images=None,
                eval=True,
                n_views=3
            )
            
            # 提取位姿
            for cam_info in scene_info.train_cameras:
                pose = np.eye(4)
                pose[:3, :3] = cam_info.R
                pose[:3, 3] = cam_info.T
                poses.append(torch.from_numpy(pose).float())
            for cam_info in scene_info.test_cameras:
                pose = np.eye(4)
                pose[:3, :3] = cam_info.R
                pose[:3, 3] = cam_info.T
                poses.append(torch.from_numpy(pose).float())

        elif data_type.lower() == 'instant':
            # 使用 readColmapSceneInfo 加载相机位姿
            print(f"从 {source_path} 加载 instant 相机位姿...")
            
            scene_info = readinstantSceneInfo(
                path=source_path,
                images=None,
                eval=True,
                n_views=3
            )
            
            # 提取位姿
            for cam_info in scene_info.train_cameras:
                pose = np.eye(4)
                pose[:3, :3] = cam_info.R
                pose[:3, 3] = cam_info.T
                poses.append(torch.from_numpy(pose).float())
            # for cam_info in scene_info.test_cameras:
            #     pose = np.eye(4)
            #     pose[:3, :3] = cam_info.R
            #     pose[:3, 3] = cam_info.T
            #     poses.append(torch.from_numpy(pose).float())

        elif data_type.lower() == 'pth':
            # 从 .pth 文件加载相机位姿
            print(f"从 {source_path} 加载 PTH 相机位姿...")
            
            pose_data = torch.load(source_path)
            if 'poses_pred' in pose_data:
                poses = pose_data['poses_pred']
            elif 'poses_gt' in pose_data:
                poses = pose_data['poses_gt']
            else:
                # 尝试直接加载 PTH 文件
                poses = pose_data
        
        elif data_type.lower() == 'txt':
            # 从文本文件加载相机位姿
            print(f"从 {source_path} 加载 TXT 相机位姿...")
            
            with open(source_path, 'r') as f:
                lines = f.readlines()
                
            for i in range(0, len(lines), 5):  # 每5行一个位姿矩阵
                if i+4 < len(lines):
                    pose = np.zeros((4, 4))
                    pose[3, 3] = 1.0
                    
                    for j in range(3):
                        values = lines[i+j].strip().split()
                        for k in range(4):
                            pose[j, k] = float(values[k])
                    
                    poses.append(torch.from_numpy(pose).float())
                
        else:
            raise ValueError(f"不支持的数据类型: {data_type}，支持的类型: dust, colmap, pth, txt")
        
        return poses
        
    def align_pose(self, pose1, pose2):
        """对齐两组位姿的位置部分"""
        mtx1 = np.array(pose1, dtype=np.double, copy=True)
        mtx2 = np.array(pose2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("输入矩阵必须是二维的")
        if mtx1.shape != mtx2.shape:
            raise ValueError("输入矩阵必须具有相同的形状")
        if mtx1.size == 0:
            raise ValueError("输入矩阵必须有>0行和>0列")

        # 将所有数据平移到原点
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("输入矩阵必须包含>1个唯一点")

        # 改变数据的缩放比例，使得trace(mtx*mtx')=1
        mtx1 /= norm1
        mtx2 /= norm2

        # 变换mtx2以最小化差异
        import scipy.linalg
        R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        mtx2 = mtx2 * s

        return mtx1, mtx2, R
        
    def evaluate(self):
        """评估位姿序列的差异并生成可视化结果"""
        # 转换为世界坐标系的位姿用于评估
        poses_a_c2w = torch.stack(self.poses_a)
        poses_b_c2w = torch.stack(self.poses_b)
        
        # 首先对位置部分进行对齐
        trans_a_align, trans_b_align, _ = self.align_pose(
            poses_a_c2w[:, :3, 3].numpy(),
            poses_b_c2w[:, :3, 3].numpy()
        )
        poses_a_aligned = poses_a_c2w.clone()
        poses_b_aligned = poses_b_c2w.clone()
        poses_a_aligned[:, :3, 3] = torch.from_numpy(trans_a_align)
        poses_b_aligned[:, :3, 3] = torch.from_numpy(trans_b_align)
        
        # 使用ATE对齐进一步优化
        poses_b_fully_aligned = align_ate_c2b_use_a2b(poses_b_aligned, poses_a_aligned)
        
        # 计算误差指标
        ate = compute_ATE(poses_a_aligned.numpy(), poses_b_fully_aligned.numpy())
        rpe_trans, rpe_rot = compute_rpe(poses_a_aligned.numpy(), poses_b_fully_aligned.numpy())
        
        # 输出评估结果
        print("\n===== 位姿评估结果 =====")
        print(f"绝对轨迹误差 (ATE): {ate:.4f}")
        print(f"相对位置误差 (RPE trans): {rpe_trans*100:.4f} cm")
        print(f"相对旋转误差 (RPE rot): {rpe_rot*180/np.pi:.4f} degrees")
        
        # 可视化轨迹
        output_path = os.path.join(self.output_dir, "poses_comparison")
        viz_path = plot_pose(
            poses_a_aligned.numpy(), 
            poses_b_fully_aligned.numpy(), 
            output_path,
            fig_size=(12, 10),
            set_limits=True
        )
        
        # 创建轨迹叠加图
        self.visualize_trajectory_overlap(
            poses_a_aligned.numpy(), 
            poses_b_fully_aligned.numpy()
        )
        
        # 保存评估结果到文件
        with open(f"{self.output_dir}/pose_evaluation.txt", 'w') as f:
            f.write("===== 位姿评估结果 =====\n")
            f.write(f"数据源A: {self.args.source_a} ({self.args.type_a})\n")
            f.write(f"数据源B: {self.args.source_b} ({self.args.type_b})\n")
            f.write(f"相机数量: {len(self.poses_a)}\n\n")
            f.write(f"绝对轨迹误差 (ATE): {ate:.4f}\n")
            f.write(f"相对位置误差 (RPE trans): {rpe_trans*100:.4f} cm\n")
            f.write(f"相对旋转误差 (RPE rot): {rpe_rot*180/np.pi:.4f} degrees\n")
        
        print(f"\n评估结果已保存到: {self.output_dir}/pose_evaluation.txt")
        print(f"轨迹可视化已保存到: {viz_path}")
        
        return {
            'ate': ate,
            'rpe_trans': rpe_trans*100,  # 转换为厘米
            'rpe_rot': rpe_rot*180/np.pi,  # 转换为角度
            'poses_a_aligned': poses_a_aligned,
            'poses_b_aligned': poses_b_fully_aligned,
            'visualization_path': viz_path
        }

    def visualize_trajectory_overlap(self, poses_a, poses_b):
        """创建叠加的轨迹可视化"""
        # 提取相机位置
        positions_a = poses_a[:, :3, 3]
        positions_b = poses_b[:, :3, 3]
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制轨迹
        # ax.plot(positions_a[:, 0], positions_a[:, 1], positions_a[:, 2], 'r-', 
        #         linewidth=2, label=f'ref pose ({self.args.type_a})')
        # ax.plot(positions_b[:, 0], positions_b[:, 1], positions_b[:, 2], 'b--', 
        #         linewidth=2, label=f'est pose ({self.args.type_b})')
        
        # 添加关键点标记（例如，每10个点标一个）
        marker_interval = max(1, len(positions_a) // 10)
        ax.scatter(positions_a[::marker_interval, 0], 
                  positions_a[::marker_interval, 1], 
                  positions_a[::marker_interval, 2], 
                  c='r', s=50, marker='o')
        ax.scatter(positions_b[::marker_interval, 0], 
                  positions_b[::marker_interval, 1], 
                  positions_b[::marker_interval, 2], 
                  c='b', s=50, marker='^')
        
        # 添加起点和终点标记
        ax.scatter(positions_a[0, 0], positions_a[0, 1], positions_a[0, 2], 
                  c='lime', s=100, marker='*', label='Start')
        ax.scatter(positions_a[-1, 0], positions_a[-1, 1], positions_a[-1, 2], 
                  c='black', s=100, marker='X', label='End')
        
        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('camera traj compare')
        
        # 设置坐标轴等比例
        max_range = np.array([
            positions_a[:, 0].max() - positions_a[:, 0].min(),
            positions_a[:, 1].max() - positions_a[:, 1].min(),
            positions_a[:, 2].max() - positions_a[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions_a[:, 0].max() + positions_a[:, 0].min()) * 0.5
        mid_y = (positions_a[:, 1].max() + positions_a[:, 1].min()) * 0.5
        mid_z = (positions_a[:, 2].max() + positions_a[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 保存图片
        output_path = os.path.join(self.output_dir, "trajectory_overlap.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"轨迹叠加图已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='简化版相机位姿评估与可视化工具')
    parser.add_argument('--source_a', type=str, required=True,
                        help='参考位姿数据源路径')
    parser.add_argument('--type_a', type=str, required=True, choices=['dust', 'colmap', 'pth', 'txt'],
                        help='参考位姿数据类型')
    parser.add_argument('--source_b', type=str, required=True,
                        help='估计位姿数据源路径')
    parser.add_argument('--type_b', type=str, required=True, choices=['dust', 'colmap', 'pth', 'txt'],
                        help='估计位姿数据类型')
    parser.add_argument('--output_dir', type=str, default='./pose_comparison_results',
                        help='输出结果目录')
    
    args = parser.parse_args()
    
    # 创建评估器并运行评估
    evaluator = CameraPoseEvaluator(args)
    evaluator.evaluate()

if __name__ == "__main__":
    main()