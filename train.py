import argparse, os
from importlib import import_module
import datetime
import subprocess

import torch

from src.util.config_parse import ConfigParser
from src.trainer import get_trainer_class


def save_training_results(trainer, config_name):
    """保存训练结果到文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f"training_results_{timestamp}.txt"
    
    with open(result_file, "w") as f:
        f.write(f"Training Results for {config_name}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Training completed for {trainer.epoch} epochs\n")
        if hasattr(trainer, 'total_training_time'):
            f.write(f"Total Training Time: {trainer.total_training_time}\n")
        
    print(f"\nResults saved to {result_file}")
    return result_file

def shutdown_system():
    print("\nTraining completed. System will shutdown in 1 minute...")
    if os.name == 'nt':  # Windows
        os.system('shutdown /s /t 60')
    else:  # Linux/Unix
        os.system('sudo shutdown -h +1')

def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-s', '--session_name', default=None,  type=str)
    args.add_argument('-c', '--config',       default=None,  type=str)
    args.add_argument('-r', '--resume',       action='store_true')
    args.add_argument('-g', '--gpu',          default=None,  type=str)
    args.add_argument(      '--thread',       default=4,     type=int)
    args.add_argument(      '--no-shutdown',  action='store_true', help="Don't shutdown after training")

    args = args.parse_args()

    assert args.config is not None, 'config file path is needed'
    if args.session_name is None:
        args.session_name = args.config # set session name to config file name

    cfg = ConfigParser(args)

    # device setting
    if cfg['gpu'] is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    # intialize trainer
    trainer = get_trainer_class(cfg['trainer'])(cfg)

    try:
        # train
        trainer.train()
        
        result_file = save_training_results(trainer, args.config)
        
        if not args.no_shutdown:
            shutdown_system()
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        result_file = save_training_results(trainer, args.config)
        print("Results saved. Not shutting down due to interruption.")
    except Exception as e:
        print(f"\nError occurred during training: {str(e)}")
        with open("training_error.txt", "w") as f:
            f.write(f"Error occurred at {datetime.datetime.now()}\n")
            f.write(str(e))
        print("Error logged in training_error.txt. Not shutting down due to error.")
        raise


if __name__ == '__main__':
    main()
