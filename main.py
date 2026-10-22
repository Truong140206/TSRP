import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path



import utils
from protocols import validate_exemplar_free_protocol
import warnings


warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


def get_args():
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == 'cifar100_hideprompt_5e':
        from configs.cifar100_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('cifar100_hideprompt_5e', help='Split-CIFAR100 HiDe-Prompt configs')
    elif config == 'imr_hideprompt_5e':
        from configs.imr_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('imr_hideprompt_5e', help='Split-ImageNet-R HiDe-Prompt configs')
    elif config == 'ima_hideprompt_5e':
        from configs.ima_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('ima_hideprompt_5e', help='Split-ImageNet-A HiDe-Prompt configs')        
    elif config == 'five_datasets_hideprompt_5e':
        from configs.five_datasets_hideprompt_5e import get_args_parser
        config_parser = subparser.add_parser('five_datasets_hideprompt_5e', help='five datasets HiDe-Prompt configs')
    elif config == 'five_datasets_lora':
        from configs.five_datasets_lora import get_args_parser
        config_parser = subparser.add_parser('five_datasets_lora', help='five datasets lora configs')             
    elif config == 'cifar100_lora':
        from configs.cifar100_lora import get_args_parser
        config_parser = subparser.add_parser('cifar100_lora', help='Split-CIFAR100 lora configs')
    elif config == 'imr_lora':
        from configs.imr_lora import get_args_parser
        config_parser = subparser.add_parser('imr_lora', help='Split-ImageNet-R lora configs')
    elif config == 'ima_lora':
        from configs.ima_lora import get_args_parser
        config_parser = subparser.add_parser('ima_lora', help='Split-ImageNet-A lora configs')
    else:
        raise NotImplementedError

    get_args_parser(config_parser)
    config_parser.add_argument('--prediction_trace_dir', default=None,
                               help='Opt-in final-stage diagnostic trace; evaluation only')
    args = parser.parse_args()
    args.config = config
    return args

def main(args):
    if getattr(args, 'prediction_trace_dir', None) and (
            not args.eval or getattr(args, 'train_inference_task_only', False)):
        raise ValueError('Prediction traces are evaluation-only; no training is allowed')
    validate_exemplar_free_protocol(args)
    utils.init_distributed_mode(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if hasattr(args, 'train_inference_task_only') and args.train_inference_task_only:
        import trainers.tii_trainer as tii_trainer
        tii_trainer.train(args)
    elif 'lora' in args.config and not args.train_inference_task_only:
        import trainers.lora_trainer as lora_trainer
        lora_trainer.train(args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    
    args = get_args()
    print(args)
    try:
        main(args)
    finally:
        utils.cleanup_distributed()
