from train.argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    parser.add_argument("--long_center", type=str, default="./train/TwDH/center/mirflickr/long")
    parser.add_argument("--short_center", type=str, default="./train/TwDH/center/mirflickr/short")
    parser.add_argument("--trans_matrix", type=str, default="./train/TwDH/center/mirflickr/trans")
    parser.add_argument("--quan_alpha", type=float, default="0.5")
    parser.add_argument("--low_rate", type=float, default="0")

    args = parser.parse_args()

    return args

