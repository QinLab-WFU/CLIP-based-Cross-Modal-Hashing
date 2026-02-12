import argparse
import os

from argsbase import get_baseargs


def get_args(main_args, trial=None):

    parser = get_baseargs()

    parser.add_argument("--n_layers", type=int, default=2, help="GNN_LAYER")
    parser.add_argument("--n_heads", type=int, default=4, help="ATT_HEAD")
    parser.add_argument("--alpha", type=int, default=5, help="loss.alpha")
    parser.add_argument("--beta", type=int, default=2, help="loss.beta")
    parser.add_argument("--lambda1", type=float, default=1, help="weight of J_r")
    parser.add_argument("--lambda2", type=float, default=1, help="weight of J_gca")
    parser.add_argument("--lambda3", type=float, default=1, help="weight of J_syn")
    parser.add_argument("--lambda4", type=float, default=10, help="weight of J_cz")
    parser.add_argument("--lambda5", type=float, default=10, help="weight of J_ce")
    parser.add_argument("--lambda6", type=float, default=10, help="weight of J_sim")
    parser.add_argument("--lambda7", type=float, default=0.3, help="weight of J_div")

    args = parser.parse_args()
    args = argparse.Namespace(**vars(args), **vars(main_args))

    if args.optuna_trail:
        args.lambda1 = trial.suggest_float('lambda1', 0.1, 2.0, log=True)
        args.lambda2 = trial.suggest_float('lambda2', 0.1, 2.0, log=True)
        args.lambda4 = trial.suggest_float('lambda4', 0.1, 20.0, log=True)
        args.lambda5 = trial.suggest_float('lambda5', 0.1, 20.0, log=True)
        args.lambda6 = trial.suggest_float('lambda6', 0.1, 20.0, log=True)
        args.lambda7 = trial.suggest_float('lambda7', 0.01, 1.0, log=True)
        args.alpha = trial.suggest_int('alpha', 1, 10)
        args.beta = trial.suggest_int('beta', 1, 10)
        args.n_layers = trial.suggest_int('n_layers', 2, 8)
        args.n_heads = trial.suggest_int('n_heads', 2, 8)
        args.lr = trial.suggest_float('lr', 1e-3, 5e-2, log=True)
        args.clip_lr = trial.suggest_float('clip_lr', 1e-5, 5e-3, log=True)

    args.method = "DGHDGH"
    flag = ""

    # flag += "noCE"
    args.noCE = True if "noCE" in flag else False

    # flag += "noDiv"
    args.div = False if "Div" in flag else True

    # flag += "noEdge"
    args.noEdge = True if "noEdge" in flag else False

    # flag += "noSim"
    args.noSim = True if "noSim" in flag else False

    args.is_optuna = False
    if args.noise_rate != 0:
        flag += f"noise{args.noise_rate}"

    if flag == "":
        args.save_dir = os.path.join(args.save_dir, args.method, args.dataset, str(args.output_dim))
    else:
        args.save_dir = os.path.join(args.save_dir, args.method, flag, args.dataset, str(args.output_dim))
    return args
