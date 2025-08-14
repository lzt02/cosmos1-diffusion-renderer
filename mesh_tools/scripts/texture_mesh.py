import os
import sys
import importlib
import click

# Add system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@click.command()
@click.option('--images_dir', '-s', 
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              required=True,
              help='Directory containing images')
@click.option('--cameras_dir', '-s', 
              type=click.Path(exists=True, file_okay=True, resolve_path=True),
              required=True,
              help='Directory containing cameras')
@click.option('--model_path', '-s',
              type=click.Path(exists=True, file_okay=False, resolve_path=True),
              required=False,
              default=None,
              help='Path to the 3D model file (OBJ format)')
@click.option('--method', '-m',
              type=click.Choice(['metashape'], case_sensitive=False),
              default='metashape',
              show_default=True,
              help='Texturing method to use')
@click.option('--texture_size',
              type=click.IntRange(1024, 8192),
              default=4096,
              show_default=True,
              help='Texture map resolution')
def main(images_dir, cameras_dir, method, texture_size, model_path=None):
    """Texture mesh using different methods"""
    # Path handling
    images_folder = images_dir
    cameras_folder = cameras_dir
        
    if not os.path.exists(images_folder) or not os.path.exists(cameras_folder):
        print(f"Error: images or cameras not found")
        print(f"Checked paths: {images_folder} and {cameras_folder}")
        return
    if model_path is not None:
        obj_file = model_path
    else:
        obj_files = [f for f in os.listdir(os.path.dirname(images_folder))
             if f.lower().endswith('.obj')]
        if obj_files:
            obj_file = os.path.join(os.path.dirname(images_folder), obj_files[0])
            
    if not os.path.exists(obj_file):
        print(f"Error: Mesh file not found at {obj_file}")
        return
    # Generate output filename
    folder_name = os.path.basename(images_folder)
    filename = f"{folder_name.split('_', 1)[-1]}.obj"
    output_dir = os.path.join(os.path.dirname(images_folder), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)

    # Check if output exists
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Exiting.")
        return

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
    
    # Run texturing
    print(f"Starting mesh texturing with {texture_size}px texture...")
    processor.texture_mesh(
        images_folder, 
        cameras_folder, 
        obj_file, 
        output_file,
        texture_size=texture_size
    )
    
    # Verify result
    if os.path.exists(output_file):
        print(f"Successfully created textured model at {output_file}")
    else:
        print(f"Error: Output file not created {output_file}")

if __name__ == "__main__":
    main()
