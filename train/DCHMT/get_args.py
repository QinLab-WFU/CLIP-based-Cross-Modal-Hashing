from train.argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    parser.add_argument("--hash-layer", type=str, default="select", help="choice a hash layer [select, linear] to run. select: select mechaism, linear: sign function.")
    parser.add_argument("--similarity-function", type=str, default="euclidean", help="choise form [cosine, euclidean]")
    parser.add_argument("--loss-type", type=str, default="l2", help="choise form [l1, l2]")

    parser.add_argument("--vartheta", type=float, default=0.5, help="the rate of error code.")
    parser.add_argument("--sim-threshold", type=float, default=0.1)

    args = parser.parse_args()

    return args

