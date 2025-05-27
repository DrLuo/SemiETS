import torch
import argparse

def convert_ckpt_semi(source_path, dest_path):

    source_weights = torch.load(source_path, map_location="cpu")['model']
    converted_weights = {}
    keys = list(source_weights.keys())

    for key in keys:
        key_s = 'student.' + key
        key_t = 'teacher.' + key
        converted_weights[key_s] = source_weights[key]
        converted_weights[key_t] = source_weights[key]

    torch.save(converted_weights, dest_path)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert pre-trained checkpoint to initialize Teacher-Student",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--source_ckpt", type=str, default="model_final.pth")
    parser.add_argument("--dest_ckpt", type=str, default="model_ts_final.pth")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    convert_ckpt_semi(args.source_ckpt, args.dest_ckpt)