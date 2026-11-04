"""
NumPy implementations of some parts of PyTorch so we can run them in a browser.
"""

import math
from contextlib import contextmanager
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np

inf = math.inf
# PyTorch keeps these behind manual_seed and no_grad.
_generator = np.random.default_rng()
_grad_enabled = True


def manual_seed(seed: int) -> None:
    global _generator
    _generator = np.random.default_rng(seed)


@contextmanager
def no_grad():
    """Build no graph inside the block, as for targets and greedy moves."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


class Tensor:
    def __init__(self, data, *, requires_grad=False, parents=(), backward_step=None):
        self.data = np.asarray(data)
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = parents
        self.backward_step = backward_step

    def __repr__(self) -> str:
        return f"tensor({self.data})"

    @property
    def shape(self):
        return self.data.shape

    def numel(self) -> int:
        return self.data.size

    def item(self):
        return self.data.item()

    def numpy(self) -> np.ndarray:
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad.")
        return self.data

    def backward(self) -> None:
        if self.data.size != 1:
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        order, visited = [], set()

        def visit(tensor):
            if id(tensor) not in visited:
                visited.add(id(tensor))
                for parent in tensor.parents:
                    visit(parent)
                order.append(tensor)

        visit(self)
        self.grad = np.ones_like(self.data)
        for tensor in reversed(order):
            if tensor.backward_step is not None and tensor.grad is not None:
                tensor.backward_step(tensor.grad)
        # a used graph is freed so second backward fails as in PyTorch
        for tensor in order:
            if tensor.backward_step is not None:
                tensor.grad = None
                tensor.backward_step = graph_freed

    def __getitem__(self, key):
        key = data_of(key)

        def backward_step(gradient):
            full = np.zeros_like(self.data)
            np.add.at(full, key, gradient)
            accumulate(self, full)

        return derived(self.data[key], (self,), backward_step)

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        index = index.data

        def backward_step(gradient):
            full = np.zeros_like(self.data)
            positions = list(np.indices(index.shape, sparse=True))
            positions[dim] = index
            np.add.at(full, tuple(positions), gradient)
            accumulate(self, full)

        return derived(np.take_along_axis(self.data, index, axis=dim), (self,), backward_step)

    def unsqueeze(self, dim: int) -> "Tensor":
        return derived(np.expand_dims(self.data, dim), (self,), self.reshaped_back)

    def squeeze(self, dim: int) -> "Tensor":
        return derived(np.squeeze(self.data, axis=dim), (self,), self.reshaped_back)

    def reshaped_back(self, gradient) -> None:
        accumulate(self, gradient.reshape(self.data.shape))

    # These have no gradient, so they refuse anything that needs one rather than quietly cut it off.

    def max(self, dim: int) -> "MaxResult":
        untracked("max", self)
        return MaxResult(Tensor(self.data.max(axis=dim)), Tensor(self.data.argmax(axis=dim)))

    def any(self, dim: int) -> "Tensor":
        untracked("any", self)
        return Tensor(self.data.any(axis=dim))

    def __invert__(self) -> "Tensor":
        untracked("~", self)
        return Tensor(~self.data)

    def __and__(self, other) -> "Tensor":
        untracked("&", self, other)
        return Tensor(self.data & data_of(other))

    def __add__(self, other) -> "Tensor":
        untracked("+", self, other)
        return Tensor(self.data + data_of(other))

    def __mul__(self, other) -> "Tensor":
        untracked("*", self, other)
        return Tensor(self.data * data_of(other))

    __radd__ = __add__
    __rmul__ = __mul__


class MaxResult(NamedTuple):
    values: Tensor
    indices: Tensor


def data_of(value):
    return value.data if isinstance(value, Tensor) else value


def derived(data, parents, backward_step) -> Tensor:
    """A tensor made from its parents, in the graph only when one of them needs a gradient."""
    if _grad_enabled and any(parent.requires_grad for parent in parents):
        return Tensor(data, requires_grad=True, parents=parents, backward_step=backward_step)
    return Tensor(data)


def accumulate(tensor: Tensor, gradient) -> None:
    if tensor.requires_grad:
        tensor.grad = gradient if tensor.grad is None else tensor.grad + gradient


def graph_freed(gradient) -> None:
    raise RuntimeError("Trying to backward through the graph a second time.")


def untracked(operation: str, *values) -> None:
    if _grad_enabled and any(isinstance(value, Tensor) and value.requires_grad for value in values):
        raise NotImplementedError(f"numpy_torch has no gradient for {operation}, so use it on values under no_grad.")


def as_tensor(data) -> Tensor:
    return data if isinstance(data, Tensor) else Tensor(data)


def where(condition, input, other) -> Tensor:
    untracked("where", input, other)
    return Tensor(np.where(data_of(condition), data_of(input), data_of(other)))


def embedding(indices, weight: Tensor) -> Tensor:
    indices = data_of(indices)

    def backward_step(gradient):
        full = np.zeros_like(weight.data)
        # a batch can hit the same room more than once, add.at adds them all up where += would drop repeats
        np.add.at(full, indices, gradient)
        accumulate(weight, full)

    return derived(weight.data[indices], (weight,), backward_step)


def linear(value: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    def backward_step(gradient):
        accumulate(value, gradient @ weight.data)
        # like PyTorch, take one unbatched input or any number of batch dimensions as rows of a single batch
        out_features, in_features = weight.shape
        gradient_rows = gradient.reshape(-1, out_features)
        accumulate(weight, gradient_rows.T @ value.data.reshape(-1, in_features))
        accumulate(bias, gradient_rows.sum(axis=0))

    return derived(value.data @ weight.data.T + bias.data, (value, weight, bias), backward_step)


def relu(value: Tensor) -> Tensor:
    def backward_step(gradient):
        accumulate(value, gradient * (value.data > 0))

    return derived(np.maximum(value.data, 0), (value,), backward_step)


def smooth_l1_loss(input: Tensor, target: Tensor, beta: float = 1.0) -> Tensor:
    """The mean Huber loss: squared for small errors, straight for big ones."""
    if beta <= 0:
        raise NotImplementedError("numpy_torch's smooth_l1_loss needs a beta above 0.")
    error = input.data - target.data
    absolute = np.abs(error)
    loss = np.where(absolute < beta, 0.5 * error**2 / beta, absolute - 0.5 * beta).mean()

    def backward_step(gradient):
        slope = np.clip(error / beta, -1, 1) * gradient / error.size
        accumulate(input, slope)
        accumulate(target, -slope)

    return derived(loss, (input, target), backward_step)


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(np.asarray(data, dtype=np.float32), requires_grad=True)


class Module:
    def __call__(self, *inputs):
        return self.forward(*inputs)

    def named_parameters(self, prefix: str = ""):
        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                yield prefix + name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{name}.")

    def parameters(self):
        return (parameter for _, parameter in self.named_parameters())

    def state_dict(self) -> dict[str, Tensor]:
        return {name: Tensor(parameter.data) for name, parameter in self.named_parameters()}

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        parameters = dict(self.named_parameters())
        if parameters.keys() != state_dict.keys():
            raise RuntimeError(f"state_dict keys {sorted(state_dict)} don't match the module's {sorted(parameters)}")
        for name, parameter in parameters.items():
            np.copyto(parameter.data, data_of(state_dict[name]))


class Sequential(Module):
    def __init__(self, *modules: Module):
        for index, module in enumerate(modules):
            setattr(self, str(index), module)

    def forward(self, value):
        for module in vars(self).values():
            value = module(value)
        return value


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = Parameter(_generator.standard_normal((num_embeddings, embedding_dim)))

    def forward(self, indices):
        return embedding(indices, self.weight)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        # PyTorch's default starting weights and biases
        bound = 1 / math.sqrt(in_features)
        self.weight = Parameter(_generator.uniform(-bound, bound, (out_features, in_features)))
        self.bias = Parameter(_generator.uniform(-bound, bound, out_features))

    def forward(self, value):
        return linear(value, self.weight, self.bias)


class ReLU(Module):
    def forward(self, value):
        return relu(value)


class Adam:
    """Adam with bias correction and PyTorch's defaults."""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.parameters = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        # as in PyTorch, each parameter counts only the steps it had a gradient for
        self.steps = [0] * len(self.parameters)
        self.averages = [np.zeros_like(parameter.data) for parameter in self.parameters]
        self.squared_averages = [np.zeros_like(parameter.data) for parameter in self.parameters]

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        # stop loudly rather than quietly fill the network with NaNs
        if not all(parameter.grad is None or np.isfinite(parameter.grad).all() for parameter in self.parameters):
            raise FloatingPointError("Training stopped on a non-finite gradient.")
        first, second = self.betas
        for index, parameter in enumerate(self.parameters):
            if parameter.grad is None:
                continue
            self.steps[index] += 1
            steps = self.steps[index]
            average, squared_average = self.averages[index], self.squared_averages[index]
            # running averages of the gradient and its square
            average *= first
            average += (1 - first) * parameter.grad
            squared_average *= second
            squared_average += (1 - second) * parameter.grad**2
            corrected = average / (1 - first**steps)
            corrected_squared = squared_average / (1 - second**steps)
            parameter.data -= self.lr * corrected / (np.sqrt(corrected_squared) + self.eps)


nn = SimpleNamespace(
    Module=Module,
    Parameter=Parameter,
    Sequential=Sequential,
    Embedding=Embedding,
    Linear=Linear,
    ReLU=ReLU,
    functional=SimpleNamespace(embedding=embedding, linear=linear, relu=relu, smooth_l1_loss=smooth_l1_loss),
)
optim = SimpleNamespace(Adam=Adam)
