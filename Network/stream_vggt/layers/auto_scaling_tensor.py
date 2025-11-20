# MIT License

# Copyright (c) 2024 Yutian Chen

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch


class AutoScalingTensor:
    def __init__(self, data: torch.Tensor, grow_dim: int = 0, curr_size: int = 0):
        self._data     = data
        self.grow_dim  = grow_dim
        self.curr_size = curr_size

    @property
    def data(self) -> torch.Tensor:
        return self._data.narrow(self.grow_dim, start=0, length=self.curr_size)
    
    @data.setter
    def data(self, value: torch.Tensor):
        self._data.narrow(self.grow_dim, start=0, length=self.curr_size).copy_(value)

    @staticmethod
    def get_scaling_factor(curr_capacity, incoming_size, factor=2):
        return max(curr_capacity * factor, curr_capacity + incoming_size)

    def __repr__(self) -> str:
        return f"AutoScalingTensor(data={self.data}, grow_on={self.grow_dim})"

    def push(self, value: torch.Tensor) -> None:
        data_size = value.size(self.grow_dim)

        # Expand the underlying tensor storage.
        if (self.curr_size + data_size) >= self._data.size(self.grow_dim):
            grow_target  = self.get_scaling_factor(self.curr_size, self.curr_size + data_size)
            # print(f"Rescaling from {self.curr_size} => {grow_target}")
            shape_target = list(self._data.shape)
            shape_target[self.grow_dim] = grow_target
            
            new_storage  = torch.empty(shape_target, dtype=self._data.dtype, device=self._data.device)
            new_storage.narrow(dim=self.grow_dim, start=0, length=self.curr_size).copy_(
                self._data.narrow(dim=self.grow_dim, start=0, length=self.curr_size)
            )
            self._data = new_storage
        
        self._data.narrow(
            dim=self.grow_dim, start=self.curr_size, length=data_size
        ).copy_(value)
        self.curr_size += data_size

    def clear(self) -> None:
        self.curr_size = 0

    def __len__(self) -> int:
        return self.curr_size

    def __getitem__(self, index) -> torch.Tensor:
        elem_selector = tuple(
            [slice(None, None, None)] * self.grow_dim + [index]
        )
        return self.data.__getitem__(elem_selector)
    
    def __setitem__(self, index, value: torch.Tensor) -> None:
        elem_selector = tuple(
            [slice(None, None, None)] * self.grow_dim + [index]
        )
        return self.data.__setitem__(elem_selector, value)

