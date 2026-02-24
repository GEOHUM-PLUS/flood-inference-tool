import argparse
import warnings
import torch

from src.inference import start_processing
from src.ui import build_ui

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ui', '--ui-mode', action='store_true', help='Activate UI mode. Ignores all other options given.')
    parser.add_argument('-i', '--input_path', type=str, help='The path to the input image containing VH and VV bands (in this order).')
    parser.add_argument('-o', '--output-path', type=str, help='The path to the final result.')
    parser.add_argument('-pp', '--post-processing', action='store_true', help='Wheter or not to apply postprocessing and reduce noise in the results.')
    parser.add_argument('-d', '--device', default='cpu', type=str, help='The device used to run the inference. Example values are "cpu", "cuda", and "mps".')
    parser.add_argument('-m', '--model', type=str, help='Model to use for the prediction.')
    parser.add_argument('-dB', action='store_true', help='Wether the SAR data is in dB.')

    args = parser.parse_args()

    if not args.ui_mode:
        if not args.input_path:
            raise ValueError('Input path must be given with -i or --input-path.')
        if not args.output_path:
            raise ValueError('Output path must be given with -o or --output-path.')

    if args.ui_mode:
        build_ui()
    else:
        try:
            torch.zeros(1).to(args.device)
            DEVICE = args.device
        except:
            warnings.warn(f'Device "{args.device}" not available, defaulting to "cpu".')
            DEVICE = 'cpu'
        start_processing(args.model, args.input_path, args.output_path, post_processing=args.post_processing, device=DEVICE, sar_is_dB=args.dB)