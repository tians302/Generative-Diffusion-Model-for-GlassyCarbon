import os
import sys
import torch

# Run from the repo root (config.yaml is read from the working directory): python tests/test_bridge.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bridge import BridgeModel, match_atoms, minimum_image, wrap, make_prior_batch
from diffusion import build_pbc_graph
from nequip_layer import NequIP_GC
from torch_geometric.data import Data

torch.manual_seed(0)
L = 14.214
CELL = torch.tensor([L, L, L])
N = 216


def random_structure(n=N):
    return torch.rand(n, 3) * CELL


def test_bridge_endpoints():
    model = BridgeModel(torch.nn.Identity(), sigma=0.5)
    x0, x1 = random_structure(), random_structure()
    cell_atom = CELL.expand(N, 3)
    d = minimum_image(x1 - x0, CELL)

    x_t, r = model.q_sample(x0, x1, torch.zeros(N, 1), cell_atom)
    assert torch.allclose(x_t, wrap(x0, CELL), atol=1e-5) # t = 0 sits on the prior
    assert torch.allclose(r, d, atol=1e-5) # and still has the full displacement to travel

    x_t, r = model.q_sample(x0, x1, torch.ones(N, 1), cell_atom)
    assert torch.allclose(minimum_image(x_t - x1, CELL), torch.zeros(N, 3), atol=1e-4) # t = 1 sits on the glass
    assert torch.allclose(r, torch.zeros(N, 3), atol=1e-6)


def test_remaining_matches_minimum_image():
    # The analytic target (1 - t) d - noise must equal minimum_image(x1 - x_t) for every intermediate t
    model = BridgeModel(torch.nn.Identity(), sigma=0.5)
    x0 = random_structure()
    x1 = wrap(x0 + 0.8 * torch.randn(N, 3), CELL) # glass-like displacements of ~1 Å
    t = torch.rand(N, 1)
    x_t, r = model.q_sample(x0, x1, t, CELL.expand(N, 3))
    assert torch.allclose(r, minimum_image(x1 - x_t, CELL), atol=1e-4)


def test_pbc_shortest_path():
    # An atom at 0.1 Å moving to L - 0.1 Å must cross the boundary (0.2 Å), not the whole box
    model = BridgeModel(torch.nn.Identity(), sigma=0.0)
    x0 = torch.tensor([[0.1, 5.0, 5.0]])
    x1 = torch.tensor([[L - 0.1, 5.0, 5.0]])
    x_t, r = model.q_sample(x0, x1, torch.full((1, 1), 0.5), CELL.expand(1, 3))
    assert abs(r[0, 0].item() + 0.1) < 1e-5
    assert min(x_t[0, 0].item(), L - x_t[0, 0].item()) < 1e-4 # midpoint lands on the boundary


def test_matching_recovers_permutation():
    x1 = random_structure()
    perm = torch.randperm(N)
    x0 = wrap(x1[perm] + 0.01 * torch.randn(N, 3), CELL)
    x0_matched, msd = match_atoms(x0, x1, CELL, random_shift=False)
    assert msd < 1e-3
    assert torch.allclose(minimum_image(x0_matched - x1, CELL), torch.zeros(N, 3), atol=0.05)
    identity_msd = (minimum_image(x0 - x1, CELL) ** 2).sum(-1).mean().item()
    assert msd < identity_msd


def _predict(model, pos):
    data = Data(pos=pos, z=torch.zeros(N, dtype=torch.long), y=torch.tensor([1.5]), cell=CELL.unsqueeze(0),
                batch=torch.zeros(N, dtype=torch.long))
    data.edge_index = build_pbc_graph(pos, CELL)
    return model(data, torch.tensor([0.3]))


def test_network_symmetries():
    torch.manual_seed(1)
    model = NequIP_GC().eval()
    pos = random_structure()
    with torch.no_grad():
        out = _predict(model, pos)
        assert out.abs().mean() > 1e-6 # guards against a trivially zero network

        # 90° rotation about z maps the cubic box onto itself, so the output must rotate with the input
        R = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        out_rot = _predict(model, wrap(pos @ R.T, CELL))
        assert torch.allclose(out_rot, out @ R.T, atol=1e-4), (out_rot - out @ R.T).abs().max()

        # Translation (with wrapping) leaves every per-atom vector unchanged
        out_shift = _predict(model, wrap(pos + torch.tensor([3.3, -7.1, 11.0]), CELL))
        assert torch.allclose(out_shift, out, atol=1e-4), (out_shift - out).abs().max()

        # Relabeling atoms relabels the outputs
        perm = torch.randperm(N)
        out_perm = _predict(model, pos[perm])
        assert torch.allclose(out_perm, out[perm], atol=1e-4)


def test_sample_runs_and_stays_in_box():
    model = BridgeModel(NequIP_GC(), sigma=0.5, num_sample_steps=3)
    priors = [random_structure(), random_structure()]
    batch = make_prior_batch(priors, [L, L, L], [0] * N, 1.5)
    x = model.sample(batch)
    assert x.shape == (2 * N, 3)
    assert torch.all(x >= 0) and torch.all(x < L)


def test_loss_backpropagates():
    model = BridgeModel(NequIP_GC(), sigma=0.5)
    from torch_geometric.data import Batch
    graphs = [Data(pos=random_structure(), pos0=random_structure(), z=torch.zeros(N, dtype=torch.long),
                   y=torch.tensor([1.5]), cell=CELL.unsqueeze(0)) for _ in range(2)]
    loss = model.get_loss(Batch.from_data_list(graphs))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"All {len(tests)} tests passed")
