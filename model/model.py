from copy import deepcopy
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from base import BaseModel

import torch
import torch.nn as nn

import numpy as np


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


class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, config, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])
        self.drop = nn.Dropout(dropout)
        self.config = config if config else None
        
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)




class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, config, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims  # [n_itmes, 600, 200]
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]  # [n_itmes, 600, 200]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]  # [n_items, 600, 400]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])  # [(n_items, 600),(600, 400)]
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])  # [(n_items, 600), (600, 200)]
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()

        self.total_anneal_steps = config['total_anneal_steps']
        self.anneal_cap = config['anneal_cap']


    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:  # 마지막 레이어가 아닐 때는 tanh를 사용함
                h = F.tanh(h)
            else:  # 마지막 레이어일 때는 결과물을 두개로 나눔
                mu = h[:, :self.q_dims[-1]]  # h[:, :200]
                logvar = h[:, self.q_dims[-1]:]  # h[:, 200:]   
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

    def get_anneal(self, update_count):
        if self.total_anneal_steps > 0:
            return min(self.anneal_cap, 1. * update_count / self.total_anneal_steps)
        else:
            return self.anneal_cap



class EASE:
    def __init__(self, _lambda):
        self.B = None
        self._lambda = _lambda
    
    def train(self, X):
        G = X.T @ X  # G = X'X
        diag_indices = list(range(G.shape[0]))
       	G[diag_indices, diag_indices] += self._lambda  # X'X + λI
        P = np.linalg.inv(G)  # P = (X'X + λI)^(-1)
        
        self.B = P / -np.diag(P)  # - P_{ij} / P_{jj} if i ≠ j
        min_dim = min(*self.B.shape)  
        self.B[range(min_dim), range(min_dim)] = 0  # 대각행렬 원소만 0으로 만들어주기 위해
        
    
    def forward(self, user_row):
        return user_row @ self.B




'''
RecVAE
'''


def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)
    

class RecVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, config):
        super(RecVAE, self).__init__()
        
        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)    
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        
        if calculate_loss:
            if gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
            negative_elbo = -(mll - kld)
            
            return (mll, kld), negative_elbo
            
        else:
            return x_pred

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))


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
        self.maniatic_embedding = nn.Linear(1, embedding_dim, bias = False)
        self.favorite_genre_embedding = nn.Embedding(18, embedding_dim, padding_idx = 0)
        self.year_embedding = nn.Embedding(88, embedding_dim, padding_idx = 0)
        self.whole_period_embedding = nn.Embedding(6, embedding_dim, padding_idx = 0)
        self.title_embedding = nn.Embedding(211, embedding_dim, padding_idx = 0)
        self.director_multihot = nn.EmbeddingBag(3654 + 1, embedding_dim, padding_idx = 0, mode = 'sum')
        self.genre_multihot = nn.EmbeddingBag(18 + 1, embedding_dim, padding_idx = 0, mode = 'sum')
        self.embedding_dim = 12 * embedding_dim

        mlp_layers = []
        for i, dim in enumerate(mlp_dims):
            if i==0:
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim))
            mlp_layers.append(nn.BatchNorm1d(dim))
            mlp_layers.append(nn.Dropout(drop_rate))
            mlp_layers.append(nn.ReLU(True))
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
        #["user_idx", "item_idx", "maniatic", "favorite_genre", "first_watch_year", "last_watch_year",
        #"whole_period", "freq_rating_year", "release_year", "series", "director", "genre"]
        user_x = x[:, :1].long()
        item_x = x[:, 1:2].long()
        maniatic_x = x[:, 2:3].float()
        favorite_genre_x = x[:, 3:4].long()
        first_watch_year_x = x[:, 4:5].long()
        last_watch_year_x = x[:, 5:6].long()
        whole_period_x = x[:, 6:7].long()
        freq_rating_year_x = x[:, 7:8].long()
        release_year_x = x[:, 8:9].long()
        title_x = x[:, 9:10].long()
        direc_x = x[:, 10:24].long()
        genre_x = x[:, 24:].long()

        user_embed_x = self.user_embedding(user_x)
        item_embed_x = self.item_embedding(item_x)
        maniatic_embed_x = self.maniatic_embedding(maniatic_x).unsqueeze(1)
        favorite_genre_embed_x = self.favorite_genre_embedding(favorite_genre_x)
        first_watch_year_embed_x = self.year_embedding(first_watch_year_x)
        last_watch_year_embed_x = self.year_embedding(last_watch_year_x)
        whole_period_embed_x = self.whole_period_embedding(whole_period_x)
        freq_rating_year_embed_x = self.year_embedding(whole_period_x)
        release_year_embed_x = self.year_embedding(release_year_x)
        title_embed_x = self.title_embedding(title_x)

        direc_embed_x = self.director_multihot(direc_x).unsqueeze(1)
        genre_embed_x = self.genre_multihot(genre_x).unsqueeze(1)

        embed_x = torch.cat([user_embed_x, item_embed_x, maniatic_embed_x, \
                            favorite_genre_embed_x, first_watch_year_embed_x, last_watch_year_embed_x, \
                            whole_period_embed_x, freq_rating_year_embed_x, release_year_embed_x, \
                            title_embed_x, direc_embed_x, genre_embed_x], dim = 1)
        
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

