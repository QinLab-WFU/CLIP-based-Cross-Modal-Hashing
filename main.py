from train.DSPH.hash_train import DSPHTrainer
from train.DCHMT.hash_train import DCHMTTrainer
from train.TwDH.hash_train import TwDHTrainer
from train.MITH.hash_train import MITHTrainer
from train.DNPH_TOMM.hash_train import DNPHTOMMTrainer
from train.DHaPH.hash_train import DHaPHTrainer
from train.DNpH_TMM.hash_train import DNpHTMMTrainer


if __name__ == "__main__":
    method = 'DNPH_TOMM'
    trainers = {
        'DSPH': DSPHTrainer,
        'DCHMT': DCHMTTrainer,
        'TwDH': TwDHTrainer,
        'MITH': MITHTrainer,
        'DNPH_TOMM': DNPHTOMMTrainer,
        'DHaPH': DHaPHTrainer,
        'DNpH_TMM': DNpHTMMTrainer
    }
    trainer = trainers.get(method)
    trainer()

