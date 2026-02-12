import argparse
import os

from argsbase import get_baseargs


def get_args(main_args):

    parser = get_baseargs()

    parser.add_argument('--use-lam', default=True, help='Use lam')
    parser.add_argument('--loss', default="acm")
    parser.add_argument('--train-n-samples', default=5)
    parser.add_argument('--valid-n-samples', default=5)
    parser.add_argument('--max-pairs', default=5000)
    parser.add_argument('--hessian-memory-factor', default=0.999, type=float, help='Dropout rate')

    parser.add_argument('--calculate-uncertainty', default=False)

    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))

    return args
