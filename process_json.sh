#!/bin/bash
dir="$1" # sourse directory
images_dir="$dir/images"
pbr_results_dir="$dir/video_results_group"

python mesh_tools/scripts/render_mesh.py --source_dir "$dir" 

bash inverse_render.sh "$dir"

python mesh_tools/scripts/align_images.py --input_dir "$pbr_results_dir" --mode "basecolor"
python mesh_tools/scripts/align_images.py --input_dir "$pbr_results_dir" --mode "roughness"
python mesh_tools/scripts/align_images.py --input_dir "$pbr_results_dir" --mode "metallic"

python mesh_tools/scripts/texture_merge.py --images_dir "$images_dir" --cameras_dir "$dir/cameras_pose.json" --method "metashape" --texture_size 4096