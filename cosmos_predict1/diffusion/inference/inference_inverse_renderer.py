# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os
import torch

from cosmos_predict1.diffusion.inference.inference_utils import add_common_arguments
from cosmos_predict1.diffusion.inference.diffusion_renderer_pipeline import DiffusionRendererPipeline
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.rendering_utils import GBUFFER_INDEX_MAPPING
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataset_inference import VideoFramesDataset
from cosmos_predict1.diffusion.inference.diffusion_renderer_utils.dataloader_utils import dict_collation_fn, dict_collation_fn_concat, sample_continuous_keys

from cosmos_predict1.utils import distributed, log, misc
from cosmos_predict1.utils.io import save_video, save_image_or_video

torch.enable_grad(False)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text to world generation demo script")
    # Add common arguments
    add_common_arguments(parser)

    # Add Diffusion Renderer specific arguments
    parser.add_argument(
        "--diffusion_transformer_dir",
        type=str,
        default="Diffusion_Renderer_Inverse_Cosmos_7B",
        help="DiT model weights directory name relative to checkpoint_dir",
        choices=[
            "Diffusion_Renderer_Inverse_Cosmos_7B",
        ],
    )

    parser.add_argument(
        "--inference_passes",
        type=str,
        default=["basecolor", "roughness", "metallic"], # ["basecolor", "normal", "depth", "roughness", "metallic"],
        nargs="+",
        help=(
            "List of G-buffer passes to infer. "
            "Options typically include: basecolor, normal, depth, roughness, metallic. "
            "Default: basecolor normal depth roughness metallic"
        )
    )
    parser.add_argument(
        "--normalize_normal",
        type=str2bool,
        default=False,
        help="If True, normal maps are normalized to unit vectors before saving. Default: False."
    )
    parser.add_argument(
        "--save_image",
        type=str2bool,
        default=True,
        help="If True, saves each output frame as an image file. Default: True."
    )
    parser.add_argument(
        "--save_video",
        type=str2bool,
        default=True,
        help="If True, saves the output as a video file. Default: True."
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to the input data. Can be a directory containing video frames or images, or a dataset name."
    )
    parser.add_argument(
        "--overlap_n_frames",
        type=int,
        default=0,
        help="Number of overlapping frames between consecutive video chunks. Useful for sliding window processing of videos."
    )
    parser.add_argument(
        "--chunk_mode",
        type=str,
        default='first',
        choices=['first', 'all', 'drop_last'],
        help="How to select video chunks from the dataset: 'first' uses only the first chunk, 'all' uses all possible chunks, 'drop_last' drops the last incomplete chunk."
    )
    parser.add_argument(
        "--group_mode",
        type=str,
        default='folder',
        choices=['folder', 'webdataset'],
        help="How to group input frames: 'folder' loads frames from a directory, 'webdataset' groups frames following the convention of webdataset."
    )
    parser.add_argument(
        "--image_extensions",
        type=str,
        default=None,
        nargs="+",
        help="List of allowed image file extensions to load (e.g., jpg, png, jpeg). If not set, uses default supported types."
    )
    parser.add_argument(
        "--resize_resolution",
        type=int,
        default=None,
        nargs="+",
        help="Resize input images to this resolution before other processing, e.g. center crop. Provide as two integers: height width. If not set, uses original image size."
    )
    parser.add_argument(
        "--ext",
        type=str,
        default=None
    )

    return parser.parse_args()


def demo(args: argparse.Namespace):
    """Run diffusion renderer inference.
    
    Args:
        args: Command line arguments
    """
    misc.set_random_seed(args.seed)

    # Initialize renderer pipeline
    pipeline = DiffusionRendererPipeline(
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.diffusion_transformer_dir,
        offload_network=args.offload_diffusion_transformer,
        offload_tokenizer=args.offload_tokenizer,
        offload_text_encoder_model=args.offload_text_encoder_model,
        offload_guardrail_models=args.offload_guardrail_models,
        guidance=args.guidance,
        num_steps=args.num_steps,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_video_frames=args.num_video_frames,
        seed=args.seed,
    )

    # Prepare input data
    dataset = VideoFramesDataset(
        root_dir=args.dataset_path,
        sample_n_frames=args.num_video_frames,
        overlap_n_frames=args.overlap_n_frames,
        chunk_mode=args.chunk_mode,
        group_mode=args.group_mode,
        image_extensions=args.image_extensions,
        resolution=[args.height, args.width],
        resize_resolution=args.resize_resolution,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=dict_collation_fn)

    # Create output directory
    os.makedirs(args.video_save_folder, exist_ok=True)

    # Generate output
    n_test = len(dataloader)
    iter_dataloader = iter(dataloader)
    print(f"Number of test samples: {n_test}")
    log.info(f"Number of test samples: {n_test}")
    for i in range(n_test):
        print('for i in range(n_test):')
        data_batch = next(iter_dataloader)

        for gbuffer_pass in args.inference_passes:
            print('for gbuffer_pass in args.inference_passes:')
            # overwrite specified G-buffer passes
            context_index = GBUFFER_INDEX_MAPPING[gbuffer_pass]
            data_batch["context_index"].fill_(context_index)

            clip_name = data_batch['clip_name'][0].replace("/", "__")
            # if os.path.exists(os.path.join(args.video_save_folder, f"{clip_name}.{gbuffer_pass}.mp4")):
            #     continue
        
            output = pipeline.generate_video(
                data_batch=data_batch,
                normalize_normal=(gbuffer_pass == 'normal' and args.normalize_normal),
            )
            print(f"{args.save_video}")
            # Save output as individual frames
            if args.save_image:
                print(f"if args.save_image:")
                video_relative_base_name = data_batch['clip_name'][0]
                chunk_ind_str = data_batch['chunk_index'][0] if 'chunk_index' in data_batch else '0000'
                for ind in range(output.shape[0]):  # (T, H, W, C)
                    idx = ind + 50*int(args.ext) if args.ext is not None else ind
                    save_path = os.path.join(args.video_save_folder, f"{gbuffer_pass}_{args.ext}", f"{idx:03d}.jpg")
                    print(args.video_save_folder)
                    print(video_relative_base_name)
                    print(f"save_path: {save_path}")
                    #return
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    print(f"Saving frame {ind} to {save_path}")
                    save_image_or_video(
                        video_save_path=save_path,
                        video=output[ind:ind + 1, ...],
                        H=args.height,
                        W=args.width,
                    )
            # Save output as video
            if args.save_video:
                print(f"if args.save_video:")
                clip_name = data_batch['clip_name'][0].replace("/", "__")
                if args.ext is not None:
                    print(f"Saving video with extension {args.ext}")
                    video_save_path = os.path.join(args.video_save_folder, f"{clip_name}.{gbuffer_pass}.{args.ext}.mp4")
                else:
                    video_save_path = os.path.join(args.video_save_folder, f"{clip_name}.{gbuffer_pass}.mp4")
                    print(f"Saving video to {video_save_path}")
                save_image_or_video(
                    video_save_path=video_save_path,
                    video=output,
                    fps=args.fps,
                    H=args.height,
                    W=args.width,
                    video_save_quality=5,
                )
                log.info(f"Saved video to {video_save_path}")


if __name__ == "__main__":
    args = parse_arguments()
    demo(args) 
