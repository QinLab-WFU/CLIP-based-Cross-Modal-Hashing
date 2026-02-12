import argparse
import os

from argsbase import get_baseargs


def get_args(main_args):

    parser = get_baseargs()
    parser.add_argument("--similarity-function", type=str, default="euclidean", help="choise from [cosine, euclidean]")

    parser.add_argument("--margin", type=float, default=0.2, help="Triplet margin.")
    parser.add_argument("--beta", default=1.2, type=float, help="Initial Class Margin Parameter in Margin Loss")

    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))
    args.save_dir = os.path.join(args.save_dir, args.method, args.loss + '_' + args.miner, str(args.output_dim) + '_')
    return args

