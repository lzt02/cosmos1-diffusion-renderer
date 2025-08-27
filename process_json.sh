#!/bin/bash
dir="$1" # sourse directory
images_dir="$dir/images"
pbr_results_dir="$dir/video_results_group"
metashape_path="path_to_your_metashape-pro"
output_dir="$dir/output"

"${metashape_path}/metashape" -r mesh_tools/scripts/render_mesh.py --source_dir "$dir" 

bash inverse_render.sh "$dir"

python mesh_tools/scripts/align_images.py --input_dir "$pbr_results_dir" --cate "basecolor" --ref_dir "$images_dir"
python mesh_tools/scripts/align_images.py --input_dir "$pbr_results_dir" --cate "roughness" --ref_dir "$images_dir"
python mesh_tools/scripts/align_images.py --input_dir "$pbr_results_dir" --cate "metallic" --ref_dir "$images_dir"

"${metashape_path}/metashape" -r mesh_tools/scripts/texture_mesh.py --images_dir "$dir/merge_basecolor_resize" --cameras_dir "$dir/camera_pose.json" --method "metashape" --texture_size 4096  -platform offscreen
"${metashape_path}/metashape" -r mesh_tools/scripts/texture_mesh.py --images_dir "$dir/merge_roughness_resize" --cameras_dir "$dir/camera_pose.json" --method "metashape" --texture_size 4096  -platform offscreen
"${metashape_path}/metashape" -r mesh_tools/scripts/texture_mesh.py --images_dir "$dir/merge_metallic_resize" --cameras_dir "$dir/camera_pose.json" --method "metashape" --texture_size 4096  -platform offscreen

find "$output_dir" -type f \( -name "roughness*.mtl" -o -name "roughness*.obj" \) -exec rm -f {} \;
find "$output_dir" -type f \( -name "metallic*.mtl" -o -name "metallic*.obj" \) -exec rm -f {} \;
