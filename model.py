import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNclass(nn.Module):
    def __init__(self, args, embed_pretrained):
        super(CNNclass, self).__init__()

        weight = torch.from_numpy(embed_pretrained).float()
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)

        # set param
        self.args = args
        self.dropout = args.dropout
        embed_dim = args.embed_dim        # 100 embed dimension
        kernel_num = args.kernel_num      # 100 filters
        kernel_sizes = args.kernel_sizes  # [3, 4, 5] 3 type of kernel sizes

        # cnn network
        self.cnn = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=kernel_num, kernel_size=size)
                                  for size in [3, 4, 5]])
        self.fc1 = nn.Linear(300, args.class_num)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x, softmax=False):
        x = self.embedding(x)   # batch x nwords x embed_dim
        x = x.permute(0, 2, 1)  # batch x embed_dim x nwords
        # conv and relu
        x = [F.relu(conv(x)) for conv in self.cnn]  # [(batch, kernel_num, nwords), ...] total 3
        # max pool
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(batch, kernel_num), ...] total 3
        # concatenate
        x = torch.cat(x, dim=1)
        x = F.dropout(x, p=self.dropout)  # batch x (3*kernel_num)
        output = self.fc1(x)              # batch x 1
        if softmax:
            output = F.softmax(output, dim=1)
        return output
