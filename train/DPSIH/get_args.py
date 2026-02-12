import argparse
import os
from argsbase import get_baseargs


def get_args(main_args):

    parser = get_baseargs()
    parser.add_argument("--use-part", default=True)
    parser.add_argument('--grad_clip', default=2.0, type=float)
    parser.add_argument('--margin', default=0.25, type=float)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--max_violation', default=True)
    parser.add_argument('--order', default=True)
    parser.add_argument('--num_embeds', default=4, type=int)
    parser.add_argument('--alpha1', default=0.01, type=float)
    parser.add_argument('--alpha2', default=0.01, type=float)

    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))
    args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))

    return args
