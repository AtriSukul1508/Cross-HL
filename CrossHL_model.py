from torch.nn import LayerNorm,Linear,Dropout,Softmax
import copy
import torch
import torch.nn as nn


class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,padding = None, bias = None,p = 64, g = 64):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.groupwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,groups=g,padding = kernel_size//3, stride = stride)
        # Pointwise Convolution
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,groups=p, stride = stride)
    def forward(self, x):
        return self.groupwise_conv(x) + self.pointwise_conv(x)






class CrossHL_attention(nn.Module):
    def __init__(self, dim, patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.1, proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.dim = dim
        self.Wq = nn.Linear(patches, dim * num_heads , bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim , bias=qkv_bias)
        self.Wv = nn.Linear(patches+1, dim , bias=qkv_bias)
        self.linear_projection = nn.Linear(dim * num_heads, dim)
        self.linear_projection_drop = nn.Dropout(proj_drop)

    def forward(self, x, x2):

        B, N, C = x.shape
        # query vector using lidar data
        query = self.Wq(x2).reshape(B, self.num_heads, self.num_heads, self.dim // self.num_heads).permute(0, 1, 2, 3)

        key = self.Wk(x).reshape(B, N, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        value = self.Wv(x.transpose(1,2)).reshape(B, C, self.num_heads, self.dim // self.num_heads).permute(0, 2, 3, 1)
        attention = torch.einsum('bhid,bhjd->bhij', key, query) * self.scale
        attention = attention.softmax(dim=-1)

        x = torch.einsum('bhij,bhjd->bhid', attention, value)
        x = x.reshape(B, N, -1)
        x = self.linear_projection(x)
        x = self.linear_projection_drop(x)
        return x

class MultiLayerPerceptron(nn.Module):
  def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fclayer1 = Linear(dim, mlp_dim)
        self.fclayer2 = Linear(mlp_dim, dim)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(0.1)
        self._init_weights()

  def _init_weights(self):
    
      nn.init.xavier_uniform_(self.fclayer1.weight)
      nn.init.xavier_uniform_(self.fclayer2.weight)

      nn.init.normal_(self.fclayer1.bias, std=1e-6)
      nn.init.normal_(self.fclayer2.bias, std=1e-6)

  def forward(self, x):
      x = self.fclayer1(x)
      x = self.act_fn(x)
      x = self.dropout(x)
      x = self.fclayer2(x)
      x = self.dropout(x)
      return x


# Code for Single Encoder block contains both cross-hl attention and mlp layer
class SingleEncoderBlock(nn.Module):
    def __init__(self,dim, num_heads, mlp_dim):
        super().__init__()
        self.attention_norm = LayerNorm(dim, eps=1e-6) # First LayerNorm layer
        self.ffn_norm = LayerNorm(dim, eps=1e-6) # Second LayerNorm layer

        self.ffn = MultiLayerPerceptron(dim, mlp_dim) # MLP layer
        self.cross_hl_attention = CrossHL_attention(dim = dim, patches = 11**2) # Cross-HL Attention layer
    def forward(self, x1,x2):
        res = x1
        x = self.attention_norm(x1)
        x= self.cross_hl_attention(x,x2)
        x = x + res
        res = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + res
        return x

# Code for Multiple Encoder blocks
class Encoder(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_dim=512, depth=2):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(dim, eps=1e-6)
        for _ in range(depth):
            layer = SingleEncoderBlock(dim, num_heads, mlp_dim)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x,x2):
        for layer_block in self.layer:
            x = layer_block(x,x2)
        encoded = self.encoder_norm(x)


        return encoded[:,0] # Selecting only the first row of each batch since it is the cls token


class CrossHL_Transformer(nn.Module):
    def __init__(self, FM, NC, NCLidar, Classes,patchsize):
        super(CrossHL_Transformer, self).__init__()
        self.patchsize = patchsize
        self.NCLidar = NCLidar
        self.conv5 = nn.Sequential(
            nn.Conv3d(1, 8, (9, 3, 3), padding=(0,1,1), stride = 1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        self.hetconv_layer = nn.Sequential(
            HetConv(8 * (NC - 8), FM*4,
                p = 1,
                g = (FM*4)//4 if (8 * (NC - 8))%FM == 0 else (FM*4)//8,
                   ),
            nn.BatchNorm2d(FM*4),
            nn.ReLU()
        )

        self.ca = Encoder(FM*4)
        self.fclayer = nn.Linear(FM*4 , Classes)
        self.position_embeddings = nn.Parameter(torch.randn(1, (patchsize**2) + 1, FM*4))
        self.dropout = nn.Dropout(0.1)
        torch.nn.init.xavier_uniform_(self.fclayer.weight)
        torch.nn.init.normal_(self.fclayer.bias, std=1e-6)
        # randomly initialized cls token
        self.clsTok = nn.Parameter(torch.zeros(1, 1, FM*4))


    def forward(self, x1, x2):
        x1 = x1.reshape(x1.shape[0],-1,self.patchsize,self.patchsize)
        x2 = x2.reshape(x1.shape[0],-1,self.patchsize*self.patchsize)
        x1 = x1.unsqueeze(1)
        if(x2.shape[1] > 0):
            x2 = F.adaptive_avg_pool1d(x2.flatten(2).transpose(1,2),1).transpose(1,2).reshape(x1.shape[0],-1,self.patchsize*self.patchsize)
        x1 = self.conv5(x1)
        x1 = x1.reshape(x1.shape[0],-1,self.patchsize,self.patchsize)

        x1 = self.hetconv_layer(x1)
        x1 = x1.flatten(2)

        x1 = x1.transpose(-1, -2)
        cls_tokens = self.clsTok.expand(x1.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x1), dim = 1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        x = self.ca(x,x2)
        x = x.reshape(x.shape[0],-1)
        out = self.fclayer(x)
        return out
