import os
import sys
import importlib
import click

# Add system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@click.command()
@click.option('--source_dir', '-s', 
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              required=True,
              help='Source directory containing images and mesh')
@click.option('--method', '-m',
              type=click.Choice(['metashape'], case_sensitive=False),
              default='metashape',
              show_default=True,
              help='Texturing method to use')
@click.option('--resolution',
              type=click.IntRange(1024, 8192),
              default=1024,
              show_default=True,
              help='Texture map resolution')
def main(source_dir, method, resolution):
    """Texture mesh using different methods"""
    # Path handling
    
    cameras_folder = source_dir
    
    obj_file = os.path.join(source_dir, "3DModel.obj")
    if not os.path.isfile(obj_file):
        # 如果没有，就找任意一个 .obj 文件
        obj_files = [f for f in os.listdir(source_dir)
             if f.lower().endswith('.obj') and f.lower().startswith('baked_')]
        if obj_files:
            obj_file = os.path.join(source_dir, obj_files[0])
        else:
            raise FileNotFoundError(f"No .obj file found in {source_dir}")

    # Dynamically load method module
    try:
        module_name = f"system.{method}"
        print(f"Loading module: {module_name}")
        module = importlib.import_module(module_name)
    except ImportError as e:
        print(f"Error: Could not import module for method {method}: {e}")
        return

    # Initialize processor
    print(f"Initializing {method} texturing processor...")
    processor = module.TextureProcessor(enable_gpu=True) 
    
    # Run rendering
    print(f"Rendering camera views for {obj_file} using {method}...")
    processor.render_camera_views(
        cameras_folder, 
        obj_file
    )


if __name__ == "__main__":
    main()