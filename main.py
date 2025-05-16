from train.DSPH.hash_train import DSPHTrainer
from train.DCHMT.hash_train import DCHMTTrainer
from train.TwDH.hash_train import TwDHTrainer
# from train.MITH.hash_train import MITHTrainer
from train.DNPH_TOMM.hash_train import DNPHTOMMTrainer
from train.DHaPH.hash_train import DHaPHTrainer
from train.DNpH_TMM.hash_train import DNpHTMMTrainer
from train.DMsH_LN.hash_train import DMsH_LNTrainer
from train.DScPH.hash_train import DScPHTrainer
from train.DDBH.hash_train import DDBHTrainer


if __name__ == "__main__":
    method = 'DDBH'
    trainers = {
        'DSPH': DSPHTrainer,
        'DCHMT': DCHMTTrainer,
        'TwDH': TwDHTrainer,
        # 'MITH': MITHTrainer,
        'DNPH_TOMM': DNPHTOMMTrainer,
        'DHaPH': DHaPHTrainer,
        'DNpH_TMM': DNpHTMMTrainer,
        'DMsH-LN': DMsH_LNTrainer,
        'DScPH': DScPHTrainer,
        'DDBH': DDBHTrainer
    }
    trainer = trainers.get(method)
    trainer()

