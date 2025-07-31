import torch

# This is the correct PyTorch implementation of the softmax and negative log likelihood loss
# that matches the micrograd version

# Create a single tensor with the same logits values as the micrograd version
logits_tensor = torch.tensor(
    [0.0, 3.0, -2.0, 1.0], dtype=torch.float64, requires_grad=True
)

# Apply softmax
probs = torch.softmax(logits_tensor, dim=0)

# Calculate negative log likelihood loss (using index 3 as the label)
loss = -torch.log(probs[3])

# Backward pass
loss.backward()

# Expected gradients from the micrograd implementation
expected_grads = [
    0.041772570515350445,
    0.8390245074625319,
    0.005653302662216329,
    -0.8864503806400986,
]

print(f"Loss value: {loss.item()}")
print(f"Expected loss: 2.1755153626167147")
print(f"Loss matches: {abs(loss.item() - 2.1755153626167147) < 1e-5}")

print("\nGradients:")
for dim in range(4):
    ok = "OK" if abs(logits_tensor.grad[dim] - expected_grads[dim]) < 1e-5 else "WRONG!"
    print(
        f"{ok} for dim {dim}: expected {expected_grads[dim]}, PyTorch returns {logits_tensor.grad[dim]}"
    )

print(
    f"\nAll gradients match: {all(abs(logits_tensor.grad[i] - expected_grads[i]) < 1e-5 for i in range(4))}"
)
