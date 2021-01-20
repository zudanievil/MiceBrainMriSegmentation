#! /usr/bin/env python3

import sys
import datetime
import pathlib

import torch

sys.path.append('.')
import unet

if __name__ == '__main__':
    # change this before training
    dataset_folder = pathlib.Path('C:/Users/user/files/lab/unet_dataset')
    save_name_stencil = 'unet_{branch}_{epoch}.pth'
    branch = 'softmax'
    epoch = 31

    # details
    num_workers = 3
    batch_size = 10
    learning_rate = 0.002
    sgd_momentum = 0.9

    # preparing loaders # todo: to function 'loader'
    loaders = {'train': None, 'test': None}
    tform = unet.RandomTransformWrapper()
    for mode in ('train', 'test'):
        ds = unet.FolderAsUnetDataset(dataset_folder / mode, transform=tform)
        loaders[mode] = torch.utils.data.dataloader.DataLoader(ds, shuffle=True,
                                    batch_size=batch_size, num_workers=num_workers)
    del ds
    print(datetime.datetime.now(), 'loaders loaded')

    # load model and optimizer state
    ckpt = torch.load(dataset_folder / save_name_stencil.format(branch=branch, epoch=epoch))
    model = unet.Unet(**ckpt['model_configuration'])
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
    loss_function = torch.nn.MSELoss()
    del ckpt
    print(datetime.datetime.now(), 'checkpoint checked')

    # training loop
    while True:
        epoch += 1
        print(datetime.datetime.now(), f'epoch {epoch} start')
        losses = {'train': list(), 'test': list()}
        for mode in ('train', 'test'):
            if mode == 'train':
                model.train()
            else:
                model.eval()
            step = 0
            for inp, gnd in loaders[mode]:
                optimizer.zero_grad()
                pred = model(inp)
                loss = loss_function(pred, gnd)
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                losses[mode].append(loss.item())
        # create checkpoint
        ckpt = {
            'model_configuration': model.config,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses,
            'epoch': epoch,
            }
        save_path = dataset_folder / save_name_stencil.format(epoch=epoch, branch=branch)
        torch.save(ckpt, save_path)
        print(datetime.datetime.now(), 'checkpoint checked out')


# # inference
# if __name__ == '__main__':
    # import numpy as np
    # from matplotlib import pyplot as plt, colors as colors

    # ds = 'test'
    # dataset_folder = pathlib.Path('C:/Users/user/files/lab/unet_dataset')
    # from_checkpoint = 'unet_softmax_23.pth'
    # #tform = unet.RandomTransformWrapper()
    # tform = None
    # ds = unet.FolderAsUnetDataset(dataset_folder / ds, transform=tform)
    # loader = torch.utils.data.dataloader.DataLoader(ds, shuffle=True)
    # bnorm = colors.BoundaryNorm(np.arange(-0.5, 3, 1), ncolors=3)
    # ckpt = torch.load(dataset_folder / from_checkpoint)
    # model = unet.Unet(**ckpt['model_configuration'])
    # model.load_state_dict(ckpt['model_state_dict'])
    # loss_fn = torch.nn.MSELoss()

    # for inp, gnd in loader:
        # pred = model(inp)
        # loss = float(loss_fn(pred, gnd).detach().numpy())
        # inp = np.squeeze(inp.detach().numpy())
        # cls = np.array([1, 2], dtype=int).reshape(2, 1, 1)
        # gnd = np.sum((gnd.detach().numpy()[0] > 0.5) * cls, axis=0)  # note that batch axis is already reduced
        # pred = np.sum((pred.detach().numpy()[0] > 0.5) * cls, axis=0)
        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        # axs = axs.flatten()
        # axs[0].imshow(inp, cmap='gray')
        # i = axs[0].imshow(gnd, cmap='Set2', alpha=0.7, norm=bnorm)
        # plt.colorbar(i, ax=axs[0], fraction=0.03)
        # axs[0].set_title('gnd')
        # axs[1].imshow(inp, cmap='gray')
        # i = axs[1].imshow(pred, cmap='Set2', alpha=0.7, norm=bnorm)
        # plt.colorbar(i, ax=axs[1], fraction=0.03)
        # axs[1].set_title(f'pred, loss: {loss}')
        # plt.show()
