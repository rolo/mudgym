"""Check the docs notebooks' NumPy stand-in for PyTorch trains a network the way PyTorch would."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

spec = importlib.util.spec_from_file_location("numpy_torch", REPOSITORY_ROOT / "docs" / "notebooks" / "numpy_torch.py")
torch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(torch)


def small_network(backend=torch, seed=0):
    backend.manual_seed(seed)
    return backend.nn.Sequential(
        backend.nn.Embedding(5, 4),
        backend.nn.Linear(4, 3),
        backend.nn.ReLU(),
        backend.nn.Linear(3, 2),
    )


def batch_loss(network, backend=torch):
    # room 2 comes up twice, so its embedding row collects two gradients
    rooms = backend.as_tensor(np.array([0, 2, 2, 4]))
    actions = backend.as_tensor(np.array([1, 0, 1, 0]))
    targets = backend.as_tensor(np.array([0.5, -2.0, 0.1, 3.0], dtype=np.float32))
    q_taken = network(rooms).gather(1, actions.unsqueeze(1)).squeeze(1)
    return backend.nn.functional.smooth_l1_loss(q_taken, targets)


def test_backward_matches_finite_differences_for_every_parameter():
    network = small_network()
    batch_loss(network).backward()

    for name, parameter in network.named_parameters():
        numeric = np.zeros(parameter.shape)
        for position in np.ndindex(parameter.shape):
            original = parameter.data[position]
            parameter.data[position] = original + 1e-3
            above = batch_loss(network).item()
            parameter.data[position] = original - 1e-3
            below = batch_loss(network).item()
            parameter.data[position] = original
            numeric[position] = (above - below) / 2e-3
        np.testing.assert_allclose(parameter.grad, numeric, atol=1e-3, err_msg=name)


def test_matches_pytorch_given_the_same_weights_and_batch():
    real_torch = pytest.importorskip("torch")
    ours = small_network()
    theirs = small_network(real_torch)
    theirs.load_state_dict({name: real_torch.tensor(parameter.data) for name, parameter in ours.named_parameters()})
    our_optimizer = torch.optim.Adam(ours.parameters(), lr=0.01)
    their_optimizer = real_torch.optim.Adam(theirs.parameters(), lr=0.01)

    # two steps, so the second runs on Adam's running averages
    for _ in range(2):
        our_loss, their_loss = batch_loss(ours), batch_loss(theirs, real_torch)
        np.testing.assert_allclose(our_loss.item(), their_loss.item(), rtol=1e-6)
        our_optimizer.zero_grad()
        their_optimizer.zero_grad()
        our_loss.backward()
        their_loss.backward()
        their_parameters = dict(theirs.named_parameters())
        for name, parameter in ours.named_parameters():
            np.testing.assert_allclose(parameter.grad, their_parameters[name].grad.numpy(), atol=1e-6, err_msg=name)
        our_optimizer.step()
        their_optimizer.step()
        for name, parameter in ours.named_parameters():
            np.testing.assert_allclose(parameter.data, their_parameters[name].detach().numpy(), atol=1e-6, err_msg=name)


def test_adam_moves_each_weight_by_the_learning_rate_against_its_gradient_on_its_first_step():
    layer = torch.nn.Linear(2, 2)
    before = layer.weight.data.copy()
    layer.weight.grad = np.array([[2.0, -3.0], [0.5, -0.25]], dtype=np.float32)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

    optimizer.step()

    np.testing.assert_allclose(before - layer.weight.data, 0.01 * np.sign(layer.weight.grad), atol=1e-6)
    optimizer.zero_grad()
    assert layer.weight.grad is None


def test_adam_counts_steps_per_parameter_so_a_late_first_gradient_still_gets_a_full_first_step():
    layer = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    for _ in range(3):
        layer.weight.grad = np.array([[1.0]], dtype=np.float32)
        optimizer.step()
    bias_before = layer.bias.data.copy()

    layer.bias.grad = np.array([1.0], dtype=np.float32)
    optimizer.step()

    np.testing.assert_allclose(bias_before - layer.bias.data, [0.01], atol=1e-6)


def test_adam_stops_on_a_non_finite_gradient_before_changing_anything():
    layer = torch.nn.Linear(1, 1)
    before = layer.weight.data.copy()
    layer.weight.grad = np.array([[np.nan]], dtype=np.float32)

    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        torch.optim.Adam(layer.parameters()).step()
    np.testing.assert_array_equal(layer.weight.data, before)


def test_a_second_backward_through_the_same_graph_fails_rather_than_doubling_the_gradients():
    network = small_network()
    loss = batch_loss(network)
    loss.backward()

    with pytest.raises(RuntimeError, match="second time"):
        loss.backward()


def test_values_made_under_no_grad_need_no_gradient_and_numpy_refuses_ones_that_do():
    layer = torch.nn.Linear(2, 1)
    value = torch.as_tensor(np.ones((1, 2), dtype=np.float32))

    tracked = layer(value)
    with torch.no_grad():
        untracked = layer(value)

    assert tracked.requires_grad
    assert not untracked.requires_grad
    np.testing.assert_array_equal(untracked.numpy(), tracked.data)
    with pytest.raises(RuntimeError, match="requires grad"):
        tracked.numpy()


def test_unsupported_uses_fail_rather_than_quietly_differ():
    tracked = torch.nn.Linear(2, 3)(torch.as_tensor(np.ones((1, 2), dtype=np.float32)))

    with pytest.raises(NotImplementedError, match="max"):
        tracked.max(dim=1)
    with pytest.raises(NotImplementedError, match="beta"):
        torch.nn.functional.smooth_l1_loss(tracked, torch.as_tensor(np.zeros((1, 3), dtype=np.float32)), beta=0)


def test_load_state_dict_copies_the_weights_without_sharing_them():
    source = small_network(seed=0)
    target = small_network(seed=1)

    target.load_state_dict(source.state_dict())
    for parameter in source.parameters():
        parameter.data += 1

    for (name, copied), (_, original) in zip(
        target.named_parameters(), small_network(seed=0).named_parameters(), strict=True
    ):
        np.testing.assert_array_equal(copied.data, original.data, err_msg=name)
