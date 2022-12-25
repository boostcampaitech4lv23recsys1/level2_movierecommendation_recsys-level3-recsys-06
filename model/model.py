import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DeepFM(BaseModel):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super().__init__()
        # input_dims = [n_users, n_items]
        # total_input_dim = int(sum(input_dims))

        # Fm component의 constant bias term과 1차 bias term
        self.bias = nn.Parameter(torch.zeros((1,)))
        self.fc = nn.Embedding(total_input_dim, 1)

        #3653 3654 17
        self.user_embedding = nn.Embedding(31360, embedding_dim)
        self.item_embedding = nn.Embedding(6807, embedding_dim)
        self.director_multihot = nn.EmbeddingBag(3654 + 1, embedding_dim, padding_idx = 0, mode = 'sum')
        self.writer_multihot = nn.EmbeddingBag(3655 + 1, embedding_dim, padding_idx = 0, mode = 'sum')
        self.genre_multihot = nn.EmbeddingBag(18 + 1, embedding_dim, padding_idx = 0, mode = 'sum')
        self.embedding_dim = 5 * embedding_dim

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim))
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input)
        
        # embed_x = self.embedding(x)

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)
        square_of_sum = torch.sum(embed_x, dim=1) ** 2 
        sum_of_square = torch.sum(embed_x ** 2, dim=1) 
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        # embed_x = self.embedding(x)
        
        #위에서 self.embedding_dim에 갯수만큼 곱해줘서 정의해둠. 
        inputs = embed_x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        # x shape: [batch_size, user(:1), item(:2), 나머지 multi hot(2:16), (16:40), (40:)]
        user_x = x[:, :1]
        item_x = x[:, 1:2]
        direc_x = x[:, 2:16]
        writer_x = x[:, 16:40]
        genre_x = x[:, 40:]
        user_embed_x = self.user_embedding(user_x)
        item_embed_x = self.item_embedding(item_x)
        direc_embed_x = self.director_multihot(direc_x)
        writer_embed_x = self.writer_multihot(writer_x)
        genre_embed_x = self.genre_multihot(genre_x)

        embed_x = torch.cat([user_embed_x, item_embed_x, direc_embed_x, writer_embed_x, genre_embed_x], dim = 1)
        
        #fm component
        fm_y = self.fm(x).squeeze(1)
        
        #deep component
        mlp_y = self.mlp(x).squeeze(1)
        
        y = torch.sigmoid(fm_y + mlp_y)
        return y

