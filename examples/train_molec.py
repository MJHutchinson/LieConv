import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.utils.utils import LoaderTo, islice, cosLr
from oil.tuning.args import argupdated_config
from oil.tuning.study import train_trial
from lie_conv.datasets import QM9datasets
from corm_data.collate import collate_fn
from lie_conv.moleculeTrainer import MolecResNet, MoleculeTrainer
from oil.datasetup.datasets import split_dataset
import lie_conv.moleculeTrainer as moleculeTrainer
import lie_conv.liegroups as liegroups
import functools
import copy


def makeTrainer(*, task='homo', device='cuda', lr=1e-2, bs=100, num_epochs=500,
                network=MolecResNet, net_config={'k':1536,'nbhd':100,'act':'swish',
        'bn':True,'aug':True,'mean':True,'num_layers':4}, recenter=False,
                subsample=False, trainer_config={'log_dir':None,'log_suffix':''}):#,'log_args':{'timeFrac':1/4,'minPeriod':0}}):
    # Create Training set and model
    device = torch.device(device)
    datasets, num_species, charge_scale = QM9datasets()
    if subsample: datasets.update(split_dataset(datasets['train'],{'train':subsample}))
    ds_stats = datasets['train'].stats[task]
    if recenter:
        m = datasets['train'].data['charges']>0
        pos = datasets['train'].data['positions'][m]
        mean,std = pos.mean(dim=0),pos.std()
        for ds in datasets.values():
            ds.data['positions'] = (ds.data['positions']-mean[None,None,:])/std
    model = network(num_species,charge_scale,**net_config).to(device)
    # if hasattr(model,'add_features_to'):
    #     for ds in datasets.values(): model.add_features_to(ds)
    # Create train and Val(Test) dataloaders and move elems to gpu
    dataloaders = {key:LoaderTo(DataLoader(dataset,batch_size=bs,num_workers=0,
                    shuffle=(key=='train'),pin_memory=True,collate_fn=collate_fn,drop_last=True),
                    device) for key,dataset in datasets.items()}
    # subsampled training dataloader for faster logging of training performance
    dataloaders['Train'] = islice(dataloaders['train'],len(dataloaders['train'])//10)
    
    # Initialize optimizer and learning rate schedule
    opt_constr = functools.partial(Adam, lr=lr)
    # if plateau:
    #     lr_sched = functools.partial(ReduceLROnPlateau,patience=20,threshold=0.02,
    #                         factor=2/3,min_lr=1e-5)
    # else:
    cos = cosLr(num_epochs)
    lr_sched = lambda e: min(e / (.01 * num_epochs), 1) * cos(e)
    return MoleculeTrainer(model,dataloaders,opt_constr,lr_sched,
                            task=task,ds_stats=ds_stats,**trainer_config)

Trial = train_trial(makeTrainer)
if __name__=='__main__':
    defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
    defaults['early_stop_metric']='valid_MAE'
    print(Trial(argupdated_config(defaults,namespace=(moleculeTrainer,liegroups))))

    # thestudy = Study(simpleTrial,argupdated_config(config_spec,namespace=__init__),
    #                 study_name="point2d",base_log_dir=log_dir)
    # thestudy.run(ordered=False)