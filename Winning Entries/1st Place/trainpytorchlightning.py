from argparse import ArgumentParser
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
import datautils
import sensorsparams
import aircraftpos

def train(plmodel,max_epochs):
    pl.seed_everything(0)
    trainer = pl.Trainer(
        gpus=1,
#        early_stop_callback=early_stop_callback,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=1,
        logger=False,
        checkpoint_callback=False,
        track_grad_norm=2,
        auto_lr_find=False,
        deterministic=True,
        weights_summary='full',
#        track_grad_norm=2,
#        val_percent_check= 0.1,
#        gradient_clip_val=plmodel.hparams.clip_grad
    )
    trainer.fit(plmodel)

dmodels = {
    'sensorsparams': sensorsparams.Test,
    'aircraftpos': aircraftpos.Test,
}
def main():
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int,default=10)
    subparsers = parser.add_subparsers(help='sub-command help', dest="model_name")
    for name, model in dmodels.items():
        subparser = subparsers.add_parser(name)
        model.add_args(subparser)
    args = parser.parse_args()
    print(args)
    Plmodel = dmodels[args.model_name]
    plmodel = Plmodel(args)
    train(plmodel,args.max_epochs)
    plmodel.after_train()

if __name__=='__main__':
    main()
