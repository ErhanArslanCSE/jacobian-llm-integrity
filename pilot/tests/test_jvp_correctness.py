"""Verify JVP matches finite differences on a small MLP.

This is a correctness test — if JVP doesn't match finite differences,
nothing downstream can be trusted.
"""

import pytest
import torch
import torch.nn as nn
from torch.func import jvp


class TwoLayerMLP(nn.Module):
    """Minimal 2-layer MLP that mimics transformer block structure."""

    def __init__(self, dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4, bias=False)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        # Residual + norm + MLP (like a transformer block)
        residual = x
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        return residual + h


class ResidualIdentityBlock(nn.Module):
    """Block where f(x) ≈ 0, so h + f(h) ≈ h, J ≈ I."""

    def __init__(self, dim=64):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        # Initialize near zero
        nn.init.normal_(self.linear.weight, std=1e-6)

    def forward(self, x):
        return x + self.linear(x)


def finite_difference_jvp(fn, x, v, eps=1e-4):
    """Compute JVP via finite differences: (f(x + eps*v) - f(x)) / eps."""
    with torch.no_grad():
        f_plus = fn(x + eps * v)
        f_base = fn(x)
    return (f_plus - f_base) / eps


class TestJVPCorrectness:
    """Core correctness tests for JVP computation."""

    def setup_method(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.dim = 64
        self.seq_len = 8

    def test_jvp_matches_finite_diff_mlp(self):
        """JVP via torch.func should match finite differences on 2-layer MLP."""
        model = TwoLayerMLP(self.dim).float()
        model.eval()

        # Input: (1, seq_len, dim)
        x = torch.randn(1, self.seq_len, self.dim)
        v = torch.randn_like(x)
        v = v / v.norm()

        # JVP via torch.func
        _, jvp_auto = jvp(model, (x,), (v,))

        # JVP via finite differences
        jvp_fd = finite_difference_jvp(model, x, v, eps=1e-4)

        # Finite differences have O(eps) error; tolerance accounts for this
        rel_error = (jvp_auto - jvp_fd).norm() / jvp_fd.norm()
        assert rel_error < 5e-2, f"JVP relative error too large: {rel_error:.6f}"

    def test_jvp_matches_finite_diff_bfloat16(self):
        """JVP should work correctly in bfloat16 (our target dtype)."""
        model = TwoLayerMLP(self.dim).bfloat16()
        model.eval()

        x = torch.randn(1, self.seq_len, self.dim, dtype=torch.bfloat16)
        v = torch.randn_like(x)
        v = v / v.norm()

        # JVP in bfloat16
        _, jvp_auto = jvp(model, (x,), (v,))

        # Finite diff in float32 as reference
        model_f32 = TwoLayerMLP(self.dim).float()
        model_f32.load_state_dict(
            {k: p.float() for k, p in model.state_dict().items()}
        )
        jvp_fd = finite_difference_jvp(model_f32, x.float(), v.float(), eps=1e-4)

        rel_error = (jvp_auto.float() - jvp_fd).norm() / jvp_fd.norm()
        # bfloat16 has less precision, allow larger tolerance
        assert rel_error < 5e-2, f"bfloat16 JVP relative error too large: {rel_error:.6f}"

    def test_zero_tangent_gives_zero_jvp(self):
        """Zero tangent vector should produce zero JVP."""
        model = TwoLayerMLP(self.dim).float()
        model.eval()

        x = torch.randn(1, self.seq_len, self.dim)
        v = torch.zeros_like(x)

        _, jvp_out = jvp(model, (x,), (v,))

        assert jvp_out.norm().item() < 1e-10, \
            f"JVP with zero tangent should be zero, got norm {jvp_out.norm().item()}"

    def test_identity_like_block_jvp_near_identity(self):
        """For h + f(h) with f ≈ 0, Jacobian ≈ I, so ||J@v|| ≈ ||v||."""
        model = ResidualIdentityBlock(self.dim).float()
        model.eval()

        x = torch.randn(1, self.seq_len, self.dim)
        v = torch.randn_like(x)
        v_norm = v.norm().item()

        _, jvp_out = jvp(model, (x,), (v,))
        jvp_norm = jvp_out.norm().item()

        # For identity-like Jacobian, ||J@v|| ≈ ||v||
        ratio = jvp_norm / v_norm
        assert 0.9 < ratio < 1.1, \
            f"Identity-like block should give ||J@v|| ≈ ||v||, got ratio {ratio:.4f}"

    def test_jvp_linearity(self):
        """JVP should be linear in the tangent: J@(av) = a * J@v."""
        model = TwoLayerMLP(self.dim).float()
        model.eval()

        x = torch.randn(1, self.seq_len, self.dim)
        v = torch.randn_like(x)
        v = v / v.norm()
        alpha = 3.7

        _, jvp_v = jvp(model, (x,), (v,))
        _, jvp_av = jvp(model, (x,), (alpha * v,))

        rel_error = (jvp_av - alpha * jvp_v).norm() / jvp_av.norm()
        assert rel_error < 1e-5, f"JVP linearity error: {rel_error:.8f}"

    def test_jvp_additivity(self):
        """JVP should be additive: J@(v1 + v2) = J@v1 + J@v2."""
        model = TwoLayerMLP(self.dim).float()
        model.eval()

        x = torch.randn(1, self.seq_len, self.dim)
        v1 = torch.randn_like(x)
        v2 = torch.randn_like(x)

        _, jvp_v1 = jvp(model, (x,), (v1,))
        _, jvp_v2 = jvp(model, (x,), (v2,))
        _, jvp_sum = jvp(model, (x,), (v1 + v2,))

        rel_error = (jvp_sum - jvp_v1 - jvp_v2).norm() / jvp_sum.norm()
        assert rel_error < 1e-5, f"JVP additivity error: {rel_error:.8f}"

    def test_last_token_tangent(self):
        """Tangent zeroed except at last token should only produce
        JVP at positions reachable from that token (causal = all positions
        since this is an MLP, not attention)."""
        model = TwoLayerMLP(self.dim).float()
        model.eval()

        x = torch.randn(1, self.seq_len, self.dim)

        # Tangent only at last position
        v = torch.zeros_like(x)
        v[0, -1, :] = torch.randn(self.dim)

        _, jvp_out = jvp(model, (x,), (v,))

        # For an MLP (no cross-position interaction), JVP should be
        # nonzero only at the last position
        other_positions_norm = jvp_out[0, :-1, :].norm().item()
        last_position_norm = jvp_out[0, -1, :].norm().item()

        assert other_positions_norm < 1e-10, \
            f"MLP JVP should be zero at non-tangent positions, got norm {other_positions_norm}"
        assert last_position_norm > 1e-6, \
            f"MLP JVP should be nonzero at tangent position, got norm {last_position_norm}"


class TestPerturbationVerification:
    """Verify that zero-epsilon perturbation gives zero difference."""

    def test_zero_epsilon_zero_diff(self):
        """Perturbation with epsilon=0 should give exactly zero differences."""
        model = TwoLayerMLP(64).float()
        model.eval()

        x = torch.randn(1, 8, 64)
        v = torch.randn_like(x)
        v = v / v.norm()

        # JVP before "perturbation"
        _, jvp_before = jvp(model, (x,), (v,))

        # Apply zero perturbation (no-op)
        for p in model.parameters():
            p.data.add_(torch.zeros_like(p.data))

        # JVP after
        _, jvp_after = jvp(model, (x,), (v,))

        diff = (jvp_after - jvp_before).norm().item()
        assert diff < 1e-10, f"Zero perturbation should give zero JVP diff, got {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
