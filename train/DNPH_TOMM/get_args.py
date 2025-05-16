from train.argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    parser.add_argument("--numclass", type=int, default=24)

    args = parser.parse_args()

    return args