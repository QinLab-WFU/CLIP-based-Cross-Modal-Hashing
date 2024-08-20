from train.argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    parser.add_argument("--numclass", type=int, default=24)
    parser.add_argument("--hypseed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.8)

    args = parser.parse_args()

    return args

