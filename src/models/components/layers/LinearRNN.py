import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Single-layer Parallel Linear RNN (with state expansion)
# ----------------------------
class ParallelLinearRNN(nn.Module):
    def __init__(self, d_model, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_model * expand

        self.dt_proj = nn.Linear(d_model, self.d_state, bias=True)
        self.b_proj = nn.Linear(d_model, self.d_state, bias=False)
        self.c_proj = nn.Linear(d_model, self.d_state, bias=False)

        self.A_log = nn.Parameter(torch.randn(self.d_state))

        # Initialize dt bias to target ~0.1
        target_dt = torch.full((self.d_state,), 0.1)
        self.dt_bias = nn.Parameter(torch.log(torch.exp(target_dt) - 1))

        self.out_proj = nn.Linear(self.d_state, d_model, bias=True)

    def discretize(self, dt, A):
        return torch.exp(A * dt)

    def forward(self, x, hidden_state=None):
        B, L, D = x.shape
        assert D == self.d_model

        dt = F.softplus(self.dt_proj(x) + self.dt_bias)  # (B, L, d_state)
        A = -torch.exp(self.A_log)                       # (d_state,)
        Bx = self.b_proj(x)                              # (B, L, d_state)
        Cx = self.c_proj(x)                              # (B, L, d_state)
        a = self.discretize(dt, A)                       # (B, L, d_state)

        if L == 1 and hidden_state is not None:
            h = a * hidden_state.unsqueeze(1) + Bx
            y = Cx * h
            y = self.out_proj(y)
            return y, h.squeeze(1)

        # Parallel scan
        a_cumprod = torch.cumprod(a, dim=1)
        weighted_Bx = Bx / (a_cumprod + 1e-12)
        h = a_cumprod * torch.cumsum(weighted_Bx, dim=1)
        y = Cx * h
        y = self.out_proj(y)

        if hidden_state is not None:
            new_h = h[:, -1, :]
            return y, new_h

        return y, None


# ----------------------------
# Single Temporal RNN Block (with residual + Norm)
# ----------------------------
class TemporalRNNBlock(nn.Module):
    def __init__(self, d_model, expand=2, dropout=0.0):
        super().__init__()
        self.rnn = ParallelLinearRNN(d_model, expand=expand)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state=None):
        """
        x: (B, T, V, P, C)
        hidden_state: (B, V, P, d_state) or None
        Returns:
            y: (B, T, V, P, C)
            new_hidden: (B, V, P, d_state) or None
        """
        B, T, V, P, C = x.shape
        d_state = self.rnn.d_state

        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * V * P, T, C).contiguous()

        h_flat = None
        if hidden_state is not None:
            assert T == 1, "Inference mode requires T=1"
            h_flat = hidden_state.reshape(B * V * P, d_state)

        y_flat, new_h_flat = self.rnn(x_flat, hidden_state=h_flat)

        y = y_flat.reshape(B, V, P, T, C).permute(0, 3, 1, 2, 4).contiguous()
        y = self.dropout(y)
        y = x + y  # residual
        y = self.norm(y)

        new_hidden = None
        if new_h_flat is not None:
            new_hidden = new_h_flat.reshape(B, V, P, d_state)

        return y, new_hidden


# ----------------------------
# Multi-layer Deep Temporal Linear RNN
# ----------------------------
class DeepTemporalLinearRNN(nn.Module):
    def __init__(self, d_model, num_layers=4, expand=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TemporalRNNBlock(d_model, expand=expand, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, hidden_states=None):
        """
        x: (B, T, V, P, C)
        hidden_states: list of [ (B, V, P, d_state) ] for each layer, or None
        Returns:
            y: (B, T, V, P, C)
            new_hidden_states: list or None
        """
        new_hidden_states = []
        for i, layer in enumerate(self.layers):
            h_in = hidden_states[i] if hidden_states is not None else None
            x, h_out = layer(x, h_in)
            new_hidden_states.append(h_out)
        return x, new_hidden_states if any(h is not None for h in new_hidden_states) else None


# ----------------------------
# Consistency verification script
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # configuration
    B, T, V, P, C = 2, 6, 3, 1041, 768
    num_layers = 1
    expand = 1

    # model
    model = DeepTemporalLinearRNN(
        d_model=C,
        num_layers=num_layers,
        expand=expand,
        dropout=0.0  # disable dropout to ensure consistency
    ).to(device)
    model.eval()

    # input
    x = torch.randn(B, T, V, P, C).to(device)

    # === parallel mode ===
    with torch.no_grad():
        y_parallel, _ = model(x)

    # === streaming inference ===
    with torch.no_grad():
        hidden_states = [torch.zeros(B, V, P, C*expand).cuda() for _ in range(num_layers)]
        y_stream_list = []
        for t in range(T):
            xt = x[:, t:t+1]  # (B, 1, V, P, C)
            yt, hidden_states = model(xt, hidden_states)
            y_stream_list.append(yt)
        y_stream = torch.cat(y_stream_list, dim=1)

    # === difference computation ===
    diff = (y_parallel - y_stream).abs().mean()
    print(f"✅ Max difference (parallel vs streaming): {diff:.2e}")

    if diff < 1e-3:
        print("🎉 Multi-layer model: parallel and streaming inference are fully consistent!")
    else:
        print("❌ Inconsistent! Please check the implementation.")

    total_params = count_parameters(model)
    print(f"✅ Total model parameters: {total_params:,}")
    print(f"   ≈ {total_params / 1e6:.2f} M (million)")