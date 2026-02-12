import argparse
import os

from argsbase import get_baseargs


def get_args(main_args):

    parser = get_baseargs()

    parser.add_argument("--HM", type=int, default=500)
    parser.add_argument("--margin", type=int, default=0.1)
    parser.add_argument("--topk", type=int, default=15)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--tau", type=int, default=0.3)

    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))
    return args
