from train.argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    args = parser.parse_args()

    return args
