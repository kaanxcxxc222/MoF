import torch
from mixture_of_flows.main import (
    FlowMatchingMoE,
    FlowMatchingMoEConfig,
)

# Define configuration
config = FlowMatchingMoEConfig(
    input_dim=512,
    hidden_dim=2048,
    num_experts=8,
    num_selected=2,
    flow_steps=10,
)

# Create model
model = FlowMatchingMoE(config)

# Forward pass
x = torch.randn(4, 128, 512)  # (batch_size, seq_len, input_dim)
output, aux_loss = model(x)

print(f"Output shape: {output.shape}")  # (4, 128, 512)
print(f"Auxiliary loss: {aux_loss.item():.6f}")
