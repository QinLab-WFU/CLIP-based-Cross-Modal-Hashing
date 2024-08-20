import argparse


def get_baseargs():

    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default="DCHMT", help='DSPH, DNPH-TOMM, DNpH-TMM, DHaPH, MITH, TwDH')
    parser.add_argument("--save-dir", type=str, default="./result/64-bit")
    parser.add_argument("--clip-path", type=str, default="./ViT-B-32.pt", help="pretrained clip path.")
    parser.add_argument("--pretrained", type=str, default="")
    parser.add_argument("--dataset", type=str, default="flickr25k", help="choise from [coco, mirflckr25k, nuswide]")
    parser.add_argument("--index-file", type=str, default="index.mat")
    parser.add_argument("--caption-file", type=str, default="caption.mat")
    parser.add_argument("--label-file", type=str, default="label.mat")

    parser.add_argument("--output-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-words", type=int, default=32)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--query-num", type=int, default=5000)
    parser.add_argument("--train-num", type=int, default=10000)
    parser.add_argument("--lr-decay-freq", type=int, default=5)
    parser.add_argument("--display-step", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1814)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--clip-lr", type=float, default=0.00001)
    parser.add_argument("--weight-decay", type=float, default=0.2)
    parser.add_argument("--warmup-proportion", type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    parser.add_argument("--is-train", action="store_true")

    return parser

