#!/bin/bash
dir="$1"  # sourse directory
images_dir="$dir/images"
tmp_base_dir="$dir/tmp_groups"
video_save_folder="$dir/video_results_group"
script_path="cosmos_predict1/diffusion/inference/inference_inverse_renderer.py"
checkpoint_dir="checkpoints"
diffusion_transformer_dir="Diffusion_Renderer_Inverse_Cosmos_7B"
dataset_path="$dir/images"
num_video_frames=57
group_mode="folder"

mkdir -p "$tmp_base_dir"
mkdir -p "$video_save_folder"

mapfile -t files < <(find "$images_dir" -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | \
    sed -E 's/.*\/([0-9]+)\.(png|jpg|jpeg)$/\1 \0/' | sort -n | awk '{print $2}')

total=${#files[@]}
group_size=57
window=50
group_idx=0

for ((start=0; start<total; start+=window)); do
        end=$((start+group_size-1))
        if (( end >= total )); then
                end=$((total-1))
        fi

        tmp_dir="$tmp_base_dir/group_$group_idx"
        mkdir -p "$tmp_dir"

        # 复制图片到临时文件夹
        for ((i=start; i<=end; i++)); do
                cp "${files[i]}" "$tmp_dir/"
        done

        # 执行推理命令
        CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python "$script_path" \
                --checkpoint_dir "$checkpoint_dir" \
                --diffusion_transformer_dir "$diffusion_transformer_dir" \
                --dataset_path="$tmp_dir" \
                --num_video_frames $num_video_frames \
                --group_mode $group_mode \
                --video_save_folder="$video_save_folder" \
		--save_image false \
                --ext "$group_idx"\
                --inference_passes "basecolor" "roughness" "metallic" 

        ((group_idx++))
done
