import torch
from torch import nn

from network.BAST import Attention


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]


class Recorder(nn.Module):
    def __init__(self, vit, device=None, share_params=False):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device
        self.share_params = share_params

    def _hook(self, m, input, output):
        # print(m)
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        if not self.share_params:
            modules1 = find_modules(self.vit.transformer1, Attention)
            modules2 = find_modules(self.vit.transformer2, Attention)
            modules3 = find_modules(self.vit.transformer3, Attention)
            modules = modules1 + modules2 + modules3
        else:
            modules1 = find_modules(self.vit.transformer1, Attention)
            modules3 = find_modules(self.vit.transformer3, Attention)
            modules = modules1 + modules3
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))
        attns = torch.stack(recordings, dim=1)
        return pred, attns


