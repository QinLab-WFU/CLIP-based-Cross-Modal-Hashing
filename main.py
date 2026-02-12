import argparse

from train.DDWSH.hash_train import DDWSHTrainer
from train.DGHDGH.hash_train import DGHDGHTrainer
from train.DPSIH.hash_train import DPSIHTrainer
from train.DSPH.hash_train import DSPHTrainer
from train.DCHMT.hash_train import DCHMTTrainer
from train.TwDH.hash_train import TwDHTrainer
from train.MITH.hash_train import MITHTrainer
from train.DNPH_TOMM.hash_train import DNPHTOMMTrainer
from train.DHaPH.hash_train import DHaPHTrainer
from train.DNpH_TMM.hash_train import DNpHTMMTrainer
from train.DMsH_LN.hash_train import DMsH_LNTrainer
from train.DScPH.hash_train import DScPHTrainer
from train.DDBH.hash_train import DDBHTrainer
from train.DPBE.hash_train import DPBETrainer

trainers = {
    'DSPH': DSPHTrainer,
    'DCHMT': DCHMTTrainer,
    'TwDH': TwDHTrainer,
    'MITH': MITHTrainer,
    'DNPH': DNPHTOMMTrainer,
    'DHaPH': DHaPHTrainer,
    'DMsH_LN': DMsH_LNTrainer,
    'DNpH': DNpHTMMTrainer,
    'DPBE': DPBETrainer,
    'DDWSH': DDWSHTrainer,
    'DDBH': DDBHTrainer,
    'DScPH': DScPHTrainer,
    'DPSIH': DPSIHTrainer,
    'DGHDGH': DGHDGHTrainer,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='DPBE', help="Trainer method name")

    parser.add_argument("--dataset", type=str, default="flickr", help="name of dataset")
    parser.add_argument("--output-dim", type=int, default=16)

    parser.add_argument("--is-train", default=True)
    args = parser.parse_args()

    trainer = trainers.get(args.method)
    trainer(args, 0)

