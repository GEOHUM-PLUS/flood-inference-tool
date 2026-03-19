import argparse
import warnings
import torch

from src.inference import start_processing
from src.ui import build_ui

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ui', '--ui-mode', action='store_true', help='Activate UI mode. Ignores all other options given.')

    parser.add_argument('-i_s1', '--input_file_sentinel_1', nargs='+', type=str, help='The path to the input image from Sentinel-1. Bands VH and VV bands (in this order).')
    parser.add_argument('-i_ps', '--input_files_planetscope', nargs='+', type=str, help='The paths to the input images from PlanetScope. Image 1: bands BGRN (in this order), Image 2: Aux with cloud mask.')
    parser.add_argument('-i_pn', '--input_files_pneo', nargs='+', type=str, help='The paths to the input images from Pleiades Neo. Image 1: bands RGB (in this order), Image 2: bands NED (in this order).')

    parser.add_argument('-o', '--output-path', type=str, help='The path to the final result.')
    parser.add_argument('-pp', '--post-processing', action='store_true', help='Wheter or not to apply postprocessing and reduce noise in the results.')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='The device used to run the inference. Example values are "cpu", "cuda", and "mps".')
    parser.add_argument('-m', '--model', type=str, help='Model to use for the prediction.')
    parser.add_argument('-dB', action='store_true', help='Wether the SAR data is in dB.')
    parser.add_argument('-bd', '--bayesian-dropout', action='store_true', help='Wether to use Bayesian Dropout to estimate uncertainty.')

    args = parser.parse_args()

    if not args.ui_mode:
        if args.input_file_sentinel_1 is None and args.input_files_planetscope is None and args.input_files_pneo is None:
            raise IOError('Please provide one of the following options: --input_file_sentinel_1, --input_files_planetscope, --input_files_pneo')
        
        if not args.input_files_planetscope is None:
            if len(args.input_files_planetscope) != 2:
                raise IOError('Please provide both Image 1 (BGRN) and Image 2 (cloud mask) only.')
        
        if not args.input_files_pneo is None:
            if len(args.input_files_pneo) != 2:
                raise IOError('Please provide both Image 1 (RGB) and Image 2 (NED) only.')
        
        if args.model is None:
            raise IOError('Please provide a model name (with extension).')

    if args.ui_mode:
        build_ui()
    else:
        try:
            torch.zeros(1).to(args.device)
            DEVICE = args.device
        except:
            warnings.warn(f'Device "{args.device}" not available, defaulting to "cpu".')
            DEVICE = 'cpu'
        start_processing(
            model_name=args.model,
            input_info={
                'input_files': args.input_file_sentinel_1 if not args.input_file_sentinel_1 is None else args.input_files_planetscope if not args.input_files_planetscope is None else args.input_files_pneo if not args.input_files_pneo is None else '',
                'data_type': 'sentinel-1' if not args.input_file_sentinel_1 is None else 'planetscope' if not args.input_files_planetscope is None else 'pleiades-neo' if not args.input_files_pneo is None else '',
                'sar_is_dB': args.dB
            },
            output_path=args.output_path,
            post_processing=args.post_processing, 
            device=DEVICE,
            bayesian_dropout=args.bayesian_dropout
        )

        # sentinel-1: [image1: VH-VV], planetscope: [image1: B-G-R-N, image2: aux], pleiades-neo: [image1: R-G-B, image2: N-E-D]