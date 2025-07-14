import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class ACRNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ACRNN, self).__init__()
        self.C = 32
        self.T = 384
        self.num_classes = num_classes

        # parameters for channel-wise attention
        self.num_channels = self.C
        self.ratio = 4  # 自定
        self.channel_wise_attention = channel_wise_attention(num_channels=self.num_channels, ratio=self.ratio)

        # parameters for CNN
        self.in_channels = 1
        self.out_channels = 40
        self.kernel_height = 32     # a
        self.kernel_width = 40     # b
        self.stride = 1
        self.pool_height = 1
        self.pool_width = 75
        self.pool_stride = 10
        self.cnn = CNN(in_channels=self.in_channels, out_channels=self.out_channels, 
                      kernel_height=self.kernel_height, kernel_width=self.kernel_width, stride=self.stride, 
                      pool_height=self.pool_height, pool_width=self.pool_width, pool_stride=self.pool_stride)
        
        # parameters for LSTM
        self.input_size = 40*1*28
        self.hidden_dim = 64
        self.lstm = LSTM(input_size=self.input_size, hidden_dim=self.hidden_dim)

        # parameters for self-attention
        self.features = self.hidden_dim
        # self.attention = MultiDimensionalAttention(feature_dim=self.features)
        self.hidden_dim_for_attention = 512     # 自定
        self.attention = ExtendedSelfAttention(input_dim=self.features, hidden_dim=self.hidden_dim_for_attention)

        # parameters for classifier
        self.classifier = nn.Linear(self.features, self.num_classes)
    
    def forward(self, x):
        # input x: (N, C, T) = (800, 32, 384)
        v = self.channel_wise_attention(x)              # v = (N, C)
        v = v.unsqueeze(2)  # v = (N, C, 1)
        c = v*x  # c = (N, C, T)
        c = c.unsqueeze(1)  # c = (N, 1, C, T)
        

        c = self.cnn(c)  # c = (N, 40, 1, 28)
        # print(f'shape after cnn: {c.shape}')
        y = self.lstm(c)  # y = (N, 1, 64)
        # print(f'shape after lstm: {y.shape}')
        y = self.attention(y)  # y = (N, 64)
        # print(f'shape after attention: {y.shape}')
        y = self.classifier(y)  # y = (N, num_classes)
        # print(f'shape after classifier: {y.shape}')
        return y
 
        


class channel_wise_attention(nn.Module):
    def __init__(self, num_channels, ratio=4):
        super(channel_wise_attention, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = ratio

        # 确保隐藏层维度至少为 1
        self.num_hidden = max(1, num_channels // ratio)

        self.fc1 = nn.Linear(num_channels, self.num_hidden)
        self.fc2 = nn.Linear(self.num_hidden, num_channels)

    # v = softmax(W2 * tanh(W1 * s_tilde + b1) + b2)
    def forward(self, x):
        # input x: (N, C, T) = (800, 32, 384)
        s_tilde = F.adaptive_avg_pool1d(x, 1)  # s_tilde = (N, C, 1)

        s_tilde = s_tilde.squeeze(2)  # s_tilde = (N, C)
        v = self.fc2(F.tanh(self.fc1(s_tilde)))  # v = (N, C)

        v = F.softmax(v, dim=-1)  
        return v
    

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_height, kernel_width, stride, 
                pool_height, pool_width, pool_stride, dropout_prob=0.5):
        super(CNN, self).__init__()
        # kernel_height = Channel = 32
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=(kernel_height, kernel_width), stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=pool_stride),
            nn.Dropout(p=dropout_prob)
        )


    def forward(self, x):
        # input x: (N, C, 1, T) = (N, 1, 32, 384)
        x = self.conv(x)    # x = (N, 40, 1, 28)
        return x
    

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_dim,
                            num_layers=2, batch_first=True)

    def forward(self, x):
        # input x: (N, 40, 1, 28)
        B = x.shape[0]
        x = x.view(B, 1, -1)  # x = (N, 1, 1120)
        y, (hidden, cell) = self.lstm(x)
        return y  # x = (N, 1, 64)


class DenseLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=nn.ELU()):
        super(DenseLayer, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.linear2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.activation = activation
        self.vector_q = nn.Linear(input_dim, input_dim, bias=True)
        self.linear_outer = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(self, h):
        q = self.vector_q(h) 
        y = self.activation(self.linear1(h) + self.linear2(q))
        z = self.linear_outer(y)
        return z


class ExtendedSelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExtendedSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.feature_wise_score_layer = DenseLayer(input_dim=self.input_dim,
                                                hidden_dim=self.hidden_dim,
                                                activation=nn.ELU()) 

        self.dropout = nn.Dropout(p=0.5) 


    def forward(self, x):
        # x = (B, 1, 64)
        h = x.squeeze(1)  # (B, 64)

        z = self.feature_wise_score_layer(h)    # (B, 64)
        
        pi = torch.sum(h*z, dim=1)  # (B,)
        pi = F.softmax(pi, dim=0) # (B,)
        pi = pi.unsqueeze(1) # (B, 1)
        
        A = pi * h # (B, 64)

        y = self.dropout(A)
        
        return y   # (B, 64)





    


    








    









# ============================================ 魔改self-attention ============================================

# def exp_mask_for_high_rank(val, mask):
#     # val: (N, 1, 64); val_mask: 1 for valid, 0 for invalid positions
#     return val + (1 - mask.float()) * -1e9

# class BnDenseLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, bias=True, activation='relu', enable_bn=True):
#         super(BnDenseLayer, self).__init__()
#         self.linear = nn.Linear(input_dim, output_dim, bias=bias)
#         self.enable_bn = enable_bn
#         if enable_bn:
#             self.bn = nn.BatchNorm1d(output_dim)
#         else:
#             self.bn = None
#         activations = {
#             'linear': nn.Identity(),
#             'relu': nn.ReLU(),
#             'elu': nn.ELU(),
#             'selu': nn.SELU(),
#         }
#         self.activation = activations.get(activation.lower(), nn.ReLU())

#     def forward(self, x):
#         # input x: (N, 1, 64)
#         batch_size, seq_len, _ = x.size()
#         x = self.linear(x)  # (N, 1, output_dim)

#         if self.enable_bn:
#             # BatchNorm1d expects (N, output_dim, 1) or (N*1, output_dim)
#             x = x.view(-1, x.size(-1))  # (N*1, output_dim)
#             x = self.bn(x)
#             x = x.view(batch_size, seq_len, -1)

#         x = self.activation(x)
#         return x

# class MultiDimensionalAttention(nn.Module):
#     def __init__(self, feature_dim, droupout_prob=0.5, activation='elu'):
#         super(MultiDimensionalAttention, self).__init__()
#         self.map1 = BnDenseLayer(feature_dim, feature_dim, bias=True, activation=activation, enable_bn=True)
#         self.map2 = BnDenseLayer(feature_dim, feature_dim, bias=True, activation='linear', enable_bn=True)
#         self.dropout = nn.Dropout(droupout_prob)

#     def forward(self, x):
#         # x: (N, 1, 64) ; 
#         map1_out = self.map1(x)  # (N, 1, 64)
#         map2_out = self.map2(map1_out)    # (N, 1, 64)

#         mask = torch.zeros_like(map2_out)    # (N, 1, 64)
#         map2_masked = exp_mask_for_high_rank(map2_out, mask)  # mask 

#         soft = F.softmax(map2_masked, dim=1)  # (N, 1, 64)

#         output = torch.sum(soft * x, dim=1)  # (N, 64)
#         output = self.dropout(output)

#         return output

    

