import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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
        # self.fc = nn.Embedding(total_input_dim, 1)
        self.fc = nn.Linear(embedding_dim, 1, bias = False)

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
        fm_y = self.bias + torch.sum(torch.sum(x, dim = 2), dim=1, keepdim = True) #64 x 64
        square_of_sum = torch.sum(x, dim=1) ** 2 #64 x 64
        sum_of_square = torch.sum(x ** 2, dim=1)# 64 x 64
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        inputs = x.view(-1, self.embedding_dim)
        mlp_y = self.mlp_layers(inputs)
        return mlp_y

    def forward(self, x):
        #하드코딩된 점을 여러 feature에 따라 수정할 수 있도록 변경하기.
        user_x = x[:, :1].long()
        item_x = x[:, 1:2].long()
        direc_x = x[:, 2:16].long()
        writer_x = x[:, 16:40].long()
        genre_x = x[:, 40:].long()
        user_embed_x = self.user_embedding(user_x)
        item_embed_x = self.item_embedding(item_x)
        direc_embed_x = self.director_multihot(direc_x).unsqueeze(1)
        writer_embed_x = self.writer_multihot(writer_x).unsqueeze(1)
        genre_embed_x = self.genre_multihot(genre_x).unsqueeze(1)

        embed_x = torch.cat([user_embed_x, item_embed_x, direc_embed_x, writer_embed_x, genre_embed_x], dim = 1)
        
        #fm component
        fm_y = self.fm(embed_x).squeeze(1)
        #deep component
        mlp_y = self.mlp(embed_x).squeeze(1)
        
        y = torch.sigmoid(fm_y + mlp_y).squeeze()
        return y


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate, num_heads):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.num_heads = num_heads
        self.d_k = self.hidden_units // self.num_heads

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.d_k)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units
        self.d_k = self.hidden_units // self.num_heads
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate, self.num_heads) # scaled dot product attention module을 사용하여 attention 계산
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        #enc = [Batch_size, seq_len, hidden_units]
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)
        
        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        # Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) 
        # K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        # V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)

        Q = self.W_Q(enc).view(batch_size, -1, self.num_heads, self.d_k) 
        K = self.W_K(enc).view(batch_size, -1, self.num_heads, self.d_k)
        V = self.W_V(enc).view(batch_size, -1, self.num_heads, self.d_k)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous() 
        output = output.view(batch_size, seqlen, -1)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        
        # SASRec과의 dimension 차이가 있습니다.
        self.W_1 = nn.Linear(hidden_units, 4 * hidden_units) 
        self.W_2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, x):
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x)))) # activation: relu -> gelu
        output = self.layerNorm(self.dropout(output) + residual)
        return output


class BERT4RecBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT4Rec(nn.Module):
    def __init__(self, device, num_user, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate):
        super(BERT4Rec, self).__init__()

        self.num_user = num_user
        self.num_item = num_item
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers 
        self.device = device
        
        self.item_emb = nn.Embedding(num_item + 2, hidden_units, padding_idx=0) # TODO2: mask와 padding을 고려하여 embedding을 생성해보세요.
        self.pos_emb = nn.Embedding(max_len, hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)
        
        self.blocks = nn.ModuleList([BERT4RecBlock(num_heads, hidden_units, dropout_rate) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_units, num_item + 1) # TODO3: 예측을 위한 output layer를 구현해보세요. (num_item 주의)

    def forward(self, log_seqs):
        seqs = self.item_emb(log_seqs)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))

        mask = torch.BoolTensor(log_seqs.detach().cpu().numpy() > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device) # mask for zero pad
        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask)
        out = self.out(seqs)
        return out