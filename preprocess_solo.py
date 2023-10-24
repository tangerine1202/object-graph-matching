import argparse
from solo_tool import Solo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to solo output')
    parser.add_argument('--output_dir', type=str, default='data', help='name of the reorganized output directory')
    parser.add_argument('--reorganized', action='store_true', help='whether the data has been reorganized')
    parser.add_argument('--move', action='store_true', help='move files instead of copying')
    args = parser.parse_args()

    solo = Solo(args.path, args.output_dir, is_reorganized=args.reorganized, move=args.move)
