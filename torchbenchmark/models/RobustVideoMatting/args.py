 import argparse

def stage1_args(start=0):
    args = parse_args()
    args.model_variant = "mobilenetv3"
    args.dataset = "videomatte"
    args.resolution_lr = 512
    args.seq_length_lr = 15
    args.learning_rate_backbone = 0.0001
    args.learning_rate_aspp = 0.0002
    args.learning_rate_decoder = 0.0002
    args.learning_rate_refiner = 0
    args.epoch_start = 0
    args.epoch_end = 1
    return args

def stage2_args():
    args = parse_args()
    args.model_variant = "mobilenetv3"
    args.dataset = "videomatte"
    args.resolution_lr = 512
    args.seq_length_lr = 50
    args.learning_rate_backbone = 0.0005
    args.learning_rate_aspp = 0.0001
    args.learning_rate_decoder = 0.0001
    args.learning_rate_refiner = 0
    args.epoch_start = 1
    args.epoch_end = 2
    return args

def stage3_args():
    args = parse_args()
    args.model_variant = "mobilenetv3"
    args.dataset = "videomatte"
    args.train_hr = True
    args.resolution_lr = 512
    args.resolution_hr = 2048
    args.seq_length_lr = 40
    args.seq_length_hr = 6
    args.learning_rate_backbone = 0.00001
    args.learning_rate_aspp = 0.00001
    args.learning_rate_decoder = 0.00001
    args.learning_rate_refiner = 0.0002
    args.epoch_start = 2
    args.epoch_end = 3
    return args

def stage4_args():
    args = parse_args()
    args.model_variant = "mobilenetv3"
    args.dataset = "imagematte"
    args.train_hr = True
    args.resolution_lr = 512
    args.resolution_hr = 2048
    args.seq_length_lr = 40
    args.seq_length_hr = 6
    args.learning_rate_backbone = 0.00001
    args.learning_rate_aspp = 0.00001
    args.learning_rate_decoder = 0.00005
    args.learning_rate_refiner = 0.0002
    args.epoch_start = 3
    args.epoch_end = 4

def parse_args(arg_string=[]):
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model-variant', type=str, required=True, choices=['mobilenetv3', 'resnet50'])
    # Matting dataset
    parser.add_argument('--dataset', type=str, required=True, choices=['videomatte', 'imagematte'])
    # Learning rate
    parser.add_argument('--learning-rate-backbone', type=float, required=True)
    parser.add_argument('--learning-rate-aspp', type=float, required=True)
    parser.add_argument('--learning-rate-decoder', type=float, required=True)
    parser.add_argument('--learning-rate-refiner', type=float, required=True)
    # Training setting
    parser.add_argument('--train-hr', action='store_true')
    parser.add_argument('--resolution-lr', type=int, default=512)
    parser.add_argument('--resolution-hr', type=int, default=2048)
    parser.add_argument('--seq-length-lr', type=int, required=True)
    parser.add_argument('--seq-length-hr', type=int, default=6)
    parser.add_argument('--downsample-ratio', type=float, default=0.25)
    parser.add_argument('--batch-size-per-gpu', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epoch-start', type=int, default=0)
    parser.add_argument('--epoch-end', type=int, default=16)
    # Tensorboard logging
    parser.add_argument('--log-dir', type=str, required=True)
    parser.add_argument('--log-train-loss-interval', type=int, default=20)
    parser.add_argument('--log-train-images-interval', type=int, default=500)
    # Checkpoint loading and saving
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--checkpoint-dir', type=str, required=True)
    parser.add_argument('--checkpoint-save-interval', type=int, default=500)
    # Distributed
    parser.add_argument('--distributed-addr', type=str, default='localhost')
    parser.add_argument('--distributed-port', type=str, default='12355')
    # Debugging
    parser.add_argument('--disable-progress-bar', action='store_true')
    parser.add_argument('--disable-validation', action='store_true')
    parser.add_argument('--disable-mixed-precision', action='store_true')
    args = parser.parse_args(arg_string)
    return args
