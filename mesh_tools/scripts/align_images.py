import os
import cv2
import numpy as np
from glob import glob
import shutil
from PIL import Image
from tqdm import tqdm


def compute_scale_shift(source, target, epsilon=1e-6):
    """计算单通道图像的scale和shift参数"""
    # 将图像展平为一维数组
    source_flat = source.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)
    
    # 计算alpha和beta（最小二乘法）
    dot_ss = np.dot(source_flat, source_flat)
    if dot_ss < epsilon:
        return 1.0, 0.0
    
    alpha = np.dot(source_flat, target_flat) / dot_ss
    beta = np.mean(target_flat - alpha * source_flat)
    return alpha, beta

def apply_transform(img, alpha, beta):
    """应用变换到图像（支持灰度和彩色）"""
    if len(img.shape) == 2:  # 灰度图
        transformed = alpha * img.astype(np.float32) + beta
    else:  # 彩色图
        transformed = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[2]):
            transformed[:, :, c] = alpha * img[:, :, c].astype(np.float32) + beta
    
    return np.clip(transformed, 0, 255).astype(np.uint8)

def get_center_region(img):
    """获取图片的中间区域（裁剪掉四周各1/4）"""
    height, width = img.shape[:2]
    
    # 计算裁剪区域
    y_start = height // 4
    y_end = height - height // 4
    x_start = width // 4
    x_end = width - width // 4
    
    # 确保索引有效
    y_start = max(0, y_start)
    y_end = min(height, y_end)
    x_start = max(0, x_start)
    x_end = min(width, x_end)
    
    if len(img.shape) == 2:  # 灰度图
        return img[y_start:y_end, x_start:x_end]
    else:  # 彩色图
        return img[y_start:y_end, x_start:x_end, :]

def align_and_merge_images(input_dir, output_dir, cate):
    """对齐并合并图片序列，使用中间区域计算对齐参数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有粗糙度文件夹（按数字排序）
    cate_dirs = sorted(glob(os.path.join(input_dir, f'{cate}_*')),
                        key=lambda x: int(os.path.basename(x).split('_')[-1]))
    
    if not cate_dirs:
        print("未找到粗糙度文件夹!")
        return
    
    print(f"找到 {len(cate_dirs)} 个粗糙度文件夹")
    print("文件夹顺序:", [os.path.basename(d) for d in cate_dirs])
    
    # 初始化计数器
    total_images = 0
    current_index = 0
    
    # 处理第一个文件夹（直接复制）
    first_dir = cate_dirs[0]
    first_images = sorted(glob(os.path.join(first_dir, '*.jpg')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    
    print(f"处理第一个文件夹: {os.path.basename(first_dir)}")
    print(f"  包含 {len(first_images)} 张图片")
    
    for img_path in first_images:
        # 复制图片到输出目录
        output_path = os.path.join(output_dir, f"{current_index:03d}.jpg")
        shutil.copy2(img_path, output_path)
        current_index += 1
        total_images += 1
    
    print(f"  已添加 {len(first_images)} 张图片")
    
    # 处理后续文件夹
    for i in range(1, len(cate_dirs)):
        current_dir = cate_dirs[i]
        print(f"\n处理文件夹: {os.path.basename(current_dir)}")
        
        # 获取当前文件夹的所有图片（按数字排序）
        current_images = sorted(glob(os.path.join(current_dir, '*.jpg')),
                            key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        
        if not current_images:
            print(f"  文件夹 {os.path.basename(current_dir)} 中没有图片!")
            continue
        
        print(f"  包含 {len(current_images)} 张图片")
        
        # 获取输出目录的最后7张图片（作为参考）
        ref_images = []
        ref_indices = range(max(0, current_index - 7), current_index)
        
        for idx in ref_indices:
            ref_path = os.path.join(output_dir, f"{idx:03d}.jpg")
            if os.path.exists(ref_path):
                ref_img = cv2.imread(ref_path)
                ref_images.append(ref_img)
            else:
                print(f"  警告: 参考图片 {ref_path} 不存在")
        
        # 获取当前文件夹的前7张图片
        new_images = []
        for j in range(min(7, len(current_images))):
            img_path = current_images[j]
            new_img = cv2.imread(img_path)
            new_images.append(new_img)
        
        # 检查是否有足够的图片进行计算
        if len(ref_images) < 7 or len(new_images) < 7:
            print(f"  警告: 没有足够的图片用于对齐计算 (参考图: {len(ref_images)}, 新图: {len(new_images)})")
            alpha, beta = 1.0, 0.0
        else:
            # 准备用于计算对齐参数的数据（只使用中间区域）
            ref_pixels = []
            new_pixels = []
            
            for j in range(7):
                # 裁剪中间区域
                ref_center = get_center_region(ref_images[j])
                new_center = get_center_region(new_images[j])
                
                # 转换为灰度图用于计算
                ref_gray = cv2.cvtColor(ref_center, cv2.COLOR_BGR2GRAY)
                new_gray = cv2.cvtColor(new_center, cv2.COLOR_BGR2GRAY)
                
                ref_pixels.append(ref_gray.flatten())
                new_pixels.append(new_gray.flatten())
            
            ref_pixels = np.concatenate(ref_pixels)
            new_pixels = np.concatenate(new_pixels)
            
            # 计算对齐参数
            alpha, beta = compute_scale_shift(new_pixels, ref_pixels)
            print(f"  对齐参数: α={alpha:.6f}, β={beta:.2f}")
            
            # 可视化中间区域（可选，用于调试）
            debug_output = os.path.join(output_dir, f"alignment_debug_{i}_{j}.jpg")
            # visualize_alignment(ref_gray, new_gray, alpha, beta, debug_output)
        
        # 处理当前文件夹的所有图片
        folder_images_added = 0
        
        for j, img_path in enumerate(current_images):
            img = cv2.imread(img_path)
            
            # 应用变换（对整个图片应用）
            # aligned_img = apply_transform(img, alpha, beta)
            aligned_img = img.astype(np.uint8)
            # 跳过前7张图片（与参考图片重叠的部分）
            if j >= 7:
                output_path = os.path.join(output_dir, f"{current_index:03d}.jpg")
                cv2.imwrite(output_path, aligned_img)
                current_index += 1
                folder_images_added += 1
                total_images += 1
        
        print(f"  已添加 {folder_images_added} 张图片")
    
    print(f"\n处理完成! 总共添加了 {total_images} 张图片到 {output_dir}")

def visualize_alignment(ref_img, new_img, alpha, beta, output_path):
    """可视化对齐效果（用于调试）"""
    # 应用变换到新图片的中间区域
    transformed = apply_transform(new_img, alpha, beta)
    
    # 创建对比图
    combined = np.hstack([ref_img, new_img, transformed])
    
    # 添加文本标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, "Reference", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Original", (ref_img.shape[1] + 10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, "Transformed", (ref_img.shape[1] * 2 + 10, 30), font, 0.7, (255, 255, 255), 2)
    
    # 添加参数信息
    param_text = f"α={alpha:.4f}, β={beta:.2f}"
    cv2.putText(combined, param_text, (10, ref_img.shape[0] - 10), font, 0.7, (255, 255, 255), 2)
    
    # 保存图像
    cv2.imwrite(output_path, combined)
    print(f"  保存对齐可视化到: {output_path}")

def resize_cate_to_fixed_size(input_dir, cate, target_size=(1024, 1024)):

    for root, dirs, files in os.walk(input_dir):
        if f"merge_{cate}" in dirs:
            cate_dir = os.path.join(root, f"merge_{cate}")
            output_dir = os.path.join(root, f"merge_{cate}_resize")

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 获取所有图片文件
            cate_files = [
                f for f in os.listdir(cate_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]

            print(f"\nProcessing folder: {root}")
            print(f"Found {len(cate_files)} images in f'merge_{cate}'")

            # 调整所有图片尺寸
            for filename in tqdm(cate_files, desc="Resizing images"):
                try:
                    input_path = os.path.join(cate_dir, filename)
                    output_path = os.path.join(output_dir, filename)

                    with Image.open(input_path) as img:
                        resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                        resized_img.save(output_path)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", "-i", type=str, required=True, 
                       help="Root directory containing subfolders with merge_*.")
    parser.add_argument("--cate", "-c", type=str, required=True, 
                       help="")
    args = parser.parse_args()

    input_directory = args.input_dir
    cate = args.cate

    output_directory = os.path.join(os.path.dirname(input_directory), f"merge_{cate}")
    
    align_and_merge_images(input_directory, output_directory, cate)
    resize_cate_to_fixed_size(os.path.dirname(input_directory), cate, target_size=(1024, 1024))