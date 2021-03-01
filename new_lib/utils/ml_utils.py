import pathlib
import datetime
from tqdm import tqdm
import numpy
import torch


def image_bhwc_to_bcwh(img: numpy.ndarray) -> numpy.ndarray:
    """convert images to torch axis order"""
    d = len(img.shape)
    if d == 2:
        img = img[numpy.newaxis, ...]
    elif d == 3:
        img = img.swapaxes(0, 2).swapaxes(0, 1)
    else:
        raise NotImplementedError(f"{d}-dimensional arrays not supported")
    return img


def image_bcwh_to_bhwc(img: numpy.ndarray) -> numpy.ndarray:
    d = len(img.shape)
    if d == 3:
        ch_d = 0
    elif d == 4:
        ch_d = 1
    else:
        raise NotImplementedError(f"{d}-dimensional arrays not supported")
    ch = img.shape[ch_d]
    if ch == 1:
        img = img.squeeze(ch_d)
    elif ch_d == 1:
        img = img.swapaxes(ch_d, ch_d+2).swapaxes(ch_d, ch_d+1)
    return img


def f_tensor(a):
    return torch.tensor(a, dtype=torch.float)


class ModelTrainer:
    """
    Wrapper for model training routine. Usage:
    path = './ckpt.pth'
    trainer = ModelTrainer(MyModel)
    trainer.restore_model(path)
    trainer.loaders['train'] = torch.utils.data.DataLoader(...)
    trainer.loaders['test'] = torch.utils.data.DataLoader(...)
    trainer.loss_function = torch.nn.MSELoss()
    trainer.optimizer = torch.optim.RMSprop(trainer.model.parameters(), lr=0.002, momentum=0.9)
    trainer.simple_train(max_epochs=-1)  # train indefinitely
    """
    def __init__(self, model_factory: callable, optimizer_factory=torch.optim.RMSprop):
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.optimizer = None
        self.loss_function = None
        self.loaders = {'train': None, 'test': None}
        self.model = None
        self._ckpt = None

    def init_model(self, checkpoint_folder: 'pathlib.Path or str', checkpoint_name: str, model_kwargs=None):
        self.model = self.model_factory(**model_kwargs) if model_kwargs else self.model_factory()
        self._ckpt = (pathlib.Path(checkpoint_folder), checkpoint_name, 0)

    def init_optimizer(self, optimizer_kwargs=None):
        self.optimizer = self.optimizer_factory(self.model.parameters(), **optimizer_kwargs)\
            if optimizer_kwargs else self.optimizer_factory(self.model.parameters())

    def restore_model(self, path, restore_optimizer_state=True):
        """resets optimizer if restore_optimizer_state=False"""
        ckpt = torch.load(path)
        self.model = self.model_factory(**ckpt['model_configuration'])
        self.model.load_state_dict(ckpt['model_state_dict'])
        self._ckpt = self._split_checkpoint_path(path)
        if restore_optimizer_state:
            self.optimizer = self.optimizer_factory(self.model.parameters())
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            self.optimizer = None

    def save_model(self, ckpt_kwargs=None):
        """saves model state and optimizer state"""
        ckpt = {
            'model_configuration': self.model.config,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if ckpt_kwargs:
            ckpt.update(ckpt_kwargs)
        f, b, e = self._ckpt
        p = f / f'{b}_{e}.pth'
        torch.save(ckpt, p)
        return p

    @staticmethod
    def _split_checkpoint_path(path: pathlib.Path):
        path = pathlib.Path(path)
        ckpt_folder = path.parent
        n = path.with_suffix('').name
        i = n.rfind('_')
        branch = n[:i]
        epoch = int(n[i+1:])
        return ckpt_folder, branch, epoch

    def epoch(self, mode: "{'train', 'test'}"):
        if mode == 'train':
            self.model.train()
        elif mode == 'test':
            self.model.eval()
        else:
            raise AssertionError
        losses = []
        for inp, gnd in tqdm(self.loaders[mode]):
            self.optimizer.zero_grad()
            pred = self.model(inp)
            loss = self.loss_function(pred, gnd)
            if mode == 'train':
                loss.backward()
                self.optimizer.step()
            losses.append(loss.item())
        return losses

    def simple_train(self, max_epochs: int = -1):
        """max_epochs < 0 means infinite number of epochs"""
        folder, branch, epoch = self._ckpt
        max_epochs += epoch
        while epoch != max_epochs:
            print(datetime.datetime.now(), 'starting epoch ', epoch)
            epoch += 1
            self._ckpt = folder, branch, epoch
            ckpt_kw = {}
            for mode in ('train', 'test'):
                ckpt_kw[mode + '_loss'] = self.epoch(mode)
            ckpt_kw['time'] = datetime.datetime.now()
            p = self.save_model(ckpt_kw)
            print('last batch train loss: ', ckpt_kw['train_loss'][-1])
            print(('test losses: ', ckpt_kw['test_loss']))
            print('saved to: ', p)


class FolderAsDataset(torch.utils.data.Dataset):
    def __init__(self, folder: pathlib.Path, transform: callable = None,
                 nn_input_key='inp', ground_truth_key='gnd'):
        self.paths = []
        for path in folder.iterdir():
            if path.suffix == '.npz':
                self.paths.append(path)
        self.transform = transform
        self.inp_k = nn_input_key
        self.gnd_k = ground_truth_key

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        x = dict(numpy.load(self.paths[idx]))
        inp = x[self.inp_k]
        gnd = x[self.gnd_k]
        if self.transform:
            inp, gnd = self.transform(inp, gnd)
        return f_tensor(inp), f_tensor(gnd)
