import torch

attn_scores = torch.tensor([[1.0, 2.0, 0.0], [4.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
print("原始分数:")
print(attn_scores)
# 创建 mask (1=保留，0=屏蔽)
mask = torch.tensor([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]])

print("\nMask (1=保留, 0=屏蔽):")
print(mask == 0)

attn_scores_masked = attn_scores.masked_fill(mask == 0, -1e-9)

print("\nMasked 后:")
print(attn_scores_masked)

# dropout 
dropout = torch.nn.Dropout(p=0.1)
torch_data = torch.tensor([[1,2,3],[3,2,1]],dtype=torch.float32)
print(torch_data)
data = dropout(torch_data)
print(data)