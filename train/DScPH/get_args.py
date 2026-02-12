import argparse
import os

from argsbase import get_baseargs


def get_args(main_args):

    parser = get_baseargs()

    parser.add_argument("--hypseed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)

    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))
    return args

