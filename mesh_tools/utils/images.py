import cv2
import numpy as np
import os
import re
import time
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms


def extract_number(filename):
    """从文件名中提取数字部分"""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"文件名中无数字部分：{filename}")

def align_saturation(input_dir, reference_dir, output_dir, group_size=5, ext=("png", "jpg", "jpeg")):
    """
    批量对齐输入目录中的图像到参考目录中的对应图像，按组计算S通道参数
    
    参数:
        input_dir (str): 输入图像目录路径
        reference_dir (str): 参考图像目录路径
        output_dir (str): 输出目录路径
        group_size (int): 组的大小，默认为5
        ext (tuple): 图像文件扩展名，默认为('png', 'jpg', 'jpeg')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    valid_exts = tuple(f".{e.lower()}" for e in ext)
    
    # 获取并配对文件
    input_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    reference_files = sorted([f for f in os.listdir(reference_dir) if f.lower().endswith(valid_exts)])
    
    if len(input_files) != len(reference_files) or \
        any(inf != ref for inf, ref in zip(input_files, reference_files)):
        raise ValueError("文件不匹配或数量不一致")
    
    # 按数字排序文件对
    file_pairs = sorted(
        [(os.path.join(input_dir, f), os.path.join(reference_dir, f)) for f in input_files],
        key=lambda x: extract_number(x[0].split(os.sep)[-1])
    )
    
    # 按组划分
    groups = [file_pairs[i:i+group_size] for i in range(0, len(file_pairs), group_size)]
    
    start_time = time.time()
    processed = 0
    
    for group_idx, group in enumerate(groups):
        # 计算组统计量
        sum_st, sum_s2, sum_s, sum_t, count = 0.0, 0.0, 0.0, 0.0, 0
        
        for inf_path, ref_path in group:
            img_input = cv2.imread(inf_path)
            img_ref = cv2.imread(ref_path)
            if img_input is None or img_ref is None:
                print(f"警告: 无法读取图像 {inf_path} 或 {ref_path}")
                continue
            
            # 转换到HSV空间并提取S通道
            s_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
            s_gt = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
            
            # 计算绝对差值并选择100%最小差值的像素
            diff = np.abs(s_input - s_gt)
            sorted_indices = np.argsort(diff.flatten())
            k = int(1.0 * diff.size)
            
            if k > 0:
                s_input_flat = s_input.flatten()
                s_gt_flat = s_gt.flatten()
                selected_indices = sorted_indices[:k]
                
                s_sel = s_input_flat[selected_indices]
                t_sel = s_gt_flat[selected_indices]
                
                # 累加统计量
                sum_st += np.sum(s_sel * t_sel)
                sum_s2 += np.sum(s_sel ** 2)
                sum_s += np.sum(s_sel)
                sum_t += np.sum(t_sel)
                count += k
        
        if count == 0:
            print(f"警告: 组 {group_idx} 无可用于计算参数的像素")
            continue
        
        # 计算线性变换参数 alpha 和 beta
        alpha = sum_st / sum_s2 if sum_s2 != 0 else 1.0
        beta = (sum_t - alpha * sum_s) / count
        
        # 应用参数到组内每个文件
        for inf_path, ref_path in group:
            img = cv2.imread(inf_path)
            if img is None:
                continue
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # 应用线性变换
            s = np.clip(alpha * s.astype(np.float32) + beta, 0, 255).astype(np.uint8)
            
            # 合并通道并保存结果
            aligned = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
            output_path = os.path.join(output_dir, os.path.basename(inf_path))
            cv2.imwrite(output_path, aligned)
            processed += 1
    
    total_time = time.time() - start_time
    print(f"处理完成: 共处理 {processed} 张图像, 耗时 {total_time:.2f}秒")
    return processed, total_time

def detail_transfer(source_image, target_image, mode="add", 
                    blur_sigma=1, blend_factor=1, mask=None, use_alpha_as_mask=True):
    """
    Transfer details from source image to target image using various blending modes.
    
    Args:
        source_image (PIL.Image): Source PIL Image
        target_image (PIL.Image): Target PIL Image
        mode (str): Blending mode - "add", "multiply", "screen", "overlay", "soft_light", 
                   "hard_light", "color_dodge", "color_burn", "difference", "exclusion", "divide"
        blur_sigma (float): Gaussian blur sigma value (0.1 to 100.0)
        blend_factor (float): Blending strength (-10.0 to 10.0)
        mask (PIL.Image or torch.Tensor, optional): External mask for selective application
        use_alpha_as_mask (bool): If True, use alpha channel from source/target as mask (default: True)
        
    Returns:
        PIL.Image: Processed PIL Image
    """
    
    def pil_to_tensor(pil_image, extract_alpha=False):
        """Convert PIL Image to tensor [1, C, H, W]"""
        # Store original mode for alpha extraction
        original_mode = pil_image.mode
        alpha_channel = None
        
        # Extract alpha channel if present and requested
        if extract_alpha and original_mode in ['RGBA', 'LA']:
            if original_mode == 'RGBA':
                r, g, b, alpha_channel = pil_image.split()
            elif original_mode == 'LA':
                l, alpha_channel = pil_image.split()
        
        # Convert to RGB for processing
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and normalize to [0, 1]
        np_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension: [H, W, C] -> [1, C, H, W]
        tensor = torch.from_numpy(np_array).permute(2, 0, 1).unsqueeze(0)
        
        # Convert alpha channel to tensor if extracted
        if alpha_channel is not None:
            alpha_array = np.array(alpha_channel).astype(np.float32) / 255.0
            alpha_tensor = torch.from_numpy(alpha_array).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            return tensor, alpha_tensor
        
        return tensor, None
    
    def tensor_to_pil(tensor, alpha_tensor=None):
        """Convert tensor [1, C, H, W] to PIL Image, optionally with alpha channel"""
        # Remove batch dimension and convert to [H, W, C]
        tensor = tensor.squeeze(0).permute(1, 2, 0)
        
        # Convert to numpy and scale to [0, 255]
        np_array = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
        
        # Create PIL Image
        if alpha_tensor is not None:
            # Add alpha channel
            alpha_array = (alpha_tensor.squeeze().numpy() * 255).clip(0, 255).astype(np.uint8)
            # Combine RGB with Alpha
            rgba_array = np.dstack([np_array, alpha_array])
            return Image.fromarray(rgba_array, mode='RGBA')
        else:
            return Image.fromarray(np_array, mode='RGB')
    
    def adjust_mask(mask, target_tensor):
        """Adjust mask dimensions to match target tensor"""
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)  # Add a channel dimension
            target_channels = target_tensor.shape[1]
            mask = mask.expand(-1, target_channels, -1, -1)
        return mask
    
    # Convert PIL Images to tensors and extract alpha channels
    target_tensor, target_alpha = pil_to_tensor(target_image, extract_alpha=use_alpha_as_mask)
    source_tensor, source_alpha = pil_to_tensor(source_image, extract_alpha=use_alpha_as_mask)
    
    # Determine device (use CPU for PIL compatibility)
    device = torch.device('cpu')
    target_tensor = target_tensor.to(device)
    source_tensor = source_tensor.to(device)
    
    # Keep track of original alpha for output
    output_alpha = target_alpha if target_alpha is not None else None
    
    B, C, H, W = target_tensor.shape
    
    # Resize source to match target dimensions if needed
    if target_tensor.shape[2:] != source_tensor.shape[2:]:
        source_tensor = F.interpolate(source_tensor, size=(H, W), mode='bilinear', align_corners=False)
    
    # Create Gaussian blur
    kernel_size = int(6 * int(blur_sigma) + 1)
    if kernel_size % 2 == 0:  # Ensure odd kernel size
        kernel_size += 1
    
    gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))
    
    # Apply blur to both images
    blurred_target = gaussian_blur(target_tensor)
    blurred_source = gaussian_blur(source_tensor)
    
    # Apply blending mode
    if mode == "add":
        tensor_out = (source_tensor - blurred_source) + blurred_target
    elif mode == "multiply":
        tensor_out = source_tensor * blurred_target
    elif mode == "screen":
        tensor_out = 1 - (1 - source_tensor) * (1 - blurred_target)
    elif mode == "overlay":
        tensor_out = torch.where(blurred_target < 0.5, 
                                2 * source_tensor * blurred_target, 
                                1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "soft_light":
        tensor_out = (1 - 2 * blurred_target) * source_tensor**2 + 2 * blurred_target * source_tensor
    elif mode == "hard_light":
        tensor_out = torch.where(source_tensor < 0.5, 
                                2 * source_tensor * blurred_target, 
                                1 - 2 * (1 - source_tensor) * (1 - blurred_target))
    elif mode == "difference":
        tensor_out = torch.abs(blurred_target - source_tensor)
    elif mode == "exclusion":
        tensor_out = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
    elif mode == "color_dodge":
        # Avoid division by zero
        tensor_out = torch.clamp(blurred_target / (1 - source_tensor + 1e-8), 0, 1)
    elif mode == "color_burn":
        # Avoid division by zero
        tensor_out = torch.clamp(1 - (1 - blurred_target) / (source_tensor + 1e-8), 0, 1)
    elif mode == "divide":
        # Avoid division by zero
        tensor_out = (source_tensor / (blurred_source + 1e-8)) * blurred_target
    else:
        tensor_out = source_tensor
    
    # Apply blend factor
    tensor_out = torch.lerp(target_tensor, tensor_out, blend_factor)
    
    # Apply mask if provided or use alpha channel
    final_mask = None
    
    # Priority: external mask > source alpha > target alpha
    if mask is not None:
        final_mask = mask
    elif use_alpha_as_mask and source_alpha is not None:
        final_mask = source_alpha
    elif use_alpha_as_mask and target_alpha is not None:
        final_mask = target_alpha
    
    if final_mask is not None:
        # Handle PIL Image mask
        if isinstance(final_mask, Image.Image):
            # Convert mask to grayscale if needed
            if final_mask.mode != 'L':
                final_mask = final_mask.convert('L')
            
            # Resize mask to match target size
            if final_mask.size != target_image.size:
                final_mask = final_mask.resize(target_image.size, Image.LANCZOS)
            
            # Convert to tensor [1, 1, H, W]
            mask_array = np.array(final_mask).astype(np.float32) / 255.0
            final_mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
        
        # Ensure mask has correct dimensions
        if final_mask is not None:
            final_mask = adjust_mask(final_mask, target_tensor)
            final_mask = final_mask.to(device)
            tensor_out = torch.lerp(target_tensor, tensor_out, final_mask)
    
    # Clamp values to valid range
    tensor_out = torch.clamp(tensor_out, 0, 1)
    
    # Convert back to PIL Image with alpha channel if original had one
    result_pil = tensor_to_pil(tensor_out, output_alpha)
    return result_pil
