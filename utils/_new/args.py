import argparse

def init_args():

    parser = argparse.ArgumentParser(
        description='Generate explanations for open-ended responses of LVLMs.'
    )

    #Hyperparams to the method
    #parser.add_argument('--L1_lambda', type=float, default=0.05)
    #parser.add_argument('--TV_lambda', type=float, default=3.0)
    parser.add_argument('--reg_lambda', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--ig_steps', type=int, default=5)
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--num_frames', type=int, default=12)
    parser.add_argument('--num_videos', type=int, default=5)
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--normalize_weights', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--randomize_data', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--use_hierarchical', type=lambda x: (str(x).lower() == 'true'), default=True)


    parser.add_argument(
        '--model',
        metavar='M',
        type=str,
        choices=['llava_video', 'qwen'],
        default='llava_video',
        help='The model to use for making predictions.')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='hd-epic',
        help='The dataset to use for making predictions.')

    parser.add_argument(
        '--output_dir',
        type=str,
        help='The path to the output directory for saving explanation results.',
        required=True)

    parser.add_argument(
        '--manual_seed',
        type=int,
        default=0,
        help='The manual seed for experiments.')

    parser.add_argument(
        '--insertion_mask_type',
        type=str,
        default='blur',
        choices=['blur', 'constant']
    )

    parser.add_argument(
        '--save_visuals', 
        type=lambda x: (str(x).lower() == 'true'), 
        default=False,
        help='Whether or not to save visuals into the folder (for large scale experiment)'
    )

    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True,
        help="The path to the input question file."
    )

    parser.add_argument(
        "--video_folder",
        type=str,
        required=True,
        help="The path to the video folder folder"
    )

    parser.add_argument(
        '--use_slic',
        type=bool,
        default=True,
        help='Use SLIC superpixels or fast 8x8 grid'
    )
    
    parser.add_argument(
        '--apply_slice',
        type=bool,
        default=False,
        help='Whether the questions define a start / end time to slice.')

    parser.add_argument(
        '--random_shuffle',
        type=bool,
        default=False,
        help='Destroy temporal order in video')

    parser.add_argument(
        '--compare_shuffle',
        type=bool,
        default=False,
        help='Do we want to compare the found heatmaps with shuffled?'
    )

    parser.add_argument(
        '--frame_analysis',
        type=bool,
        default=False,
    )
    
    parser.add_argument(
        '--mc_samples',
        type=int,
        default=50,
        help='How much samples we want to compute the MC Shapley values with'
    )

    parser.add_argument(
        '--gt_forcing',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help='Whether to run the XAI explanation using the ground truth answer (Teacher Forcing).'
    )

    # LVLM generation settings
    parser.add_argument("--temperature", type=float,default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--use_yake", type=bool, default=False, help="Whether use yake to detect keywords")
    parser.add_argument("--choices", type=bool, default=False, help="Whether ask the LVLMs to generate single choice instead of open-ended responses")
    parser.add_argument("--ablation_zero", type=bool, default=False, help="Ablation of baseline image using all-zero images")
    parser.add_argument("--ablation_noise", type=bool, default=False, help="Ablation of baseline image using random noise")
    
    #IGOS
    parser.add_argument(
        '--method',
        type=str,
        default='spix',
        help='Which explanation method to run.'
    )
    parser.add_argument('--size', type=int, default=28, help='Grid size for iGOS mask.')
    parser.add_argument('--momentum', type=int, default=3, help='NAG momentum.')
    return parser.parse_args()