from train.argsbase import get_baseargs


def get_args():

    parser = get_baseargs()

    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--top-k-label", type=int, default=8)
    parser.add_argument("--res-mlp-layers", type=int, default=2)
    parser.add_argument("--hyper-lambda", type=float, default=0.99)
    parser.add_argument("--hyper-tokens-intra", type=float, default=1)
    parser.add_argument("--hyper-cls-inter", type=float, default=10)
    parser.add_argument("--hyper-quan", type=float, default=8)
    parser.add_argument("--hyper-info-nce", type=float, default=50)
    parser.add_argument("--hyper-alpha", type=float, default=0.01)
    parser.add_argument("--hyper-distill", type=float, default=1)

    args = parser.parse_args()

    return args

