if __name__ == '__main__':
    import sys
    import pathlib
    import torch
    sys.path.append('..')
    from new_lib.core import loc_net as ln
    from new_lib.utils import ml_utils as mlu
    
    checkpoint_path = pathlib.Path("C:\\Users\\user\\files\\lab\\ds\\loc_net_hemisph\\1ch_lbbox_4.pth")
    ds_folder = pathlib.Path("C:\\Users\\user\\files\\lab\\ds\\loc_net_hemisph")
    ground_truth_key = 'gnd_l'
    # train the network
    trainer = mlu.ModelTrainer(ln.LocalizationNetwork, torch.optim.RMSprop)

    # first time initialization
    # trainer.init_model(ds_folder, '1ch_lbbox')
    # trainer.init_optimizer(dict(lr=0.002, momentum=0.9))
    # trainer.save_model()

    # restoration
    trainer.restore_model(checkpoint_path)
    # training
    trainer.loss_function = torch.nn.MSELoss(reduction='sum')
    for mode in ('train', 'test'):
        trainer.loaders[mode] = torch.utils.data.DataLoader(
            mlu.FolderAsDataset(ds_folder/mode, transform=ln.LocNetDatasetTransform(),
                                ground_truth_key=ground_truth_key),
            batch_size=10, shuffle=True, num_workers=3)
    trainer.simple_train()
