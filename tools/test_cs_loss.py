from mmseg.models.losses import CrossEntropyLoss
import torch

loss_fn = CrossEntropyLoss(class_weight=[0.2, 1.0])
pred = torch.randn(2, 2, 32, 32, requires_grad=True)
gt = torch.randint(0, 2, (2, 32, 32))

loss = loss_fn(pred, gt)
loss.backward()
print("✅ Loss OK:", loss.item())
