import torch
from torch import nn
import torch.nn.functional as F
import math
import copy
from torch.nn.parameter import Parameter

class MVA(nn.Module):
    def __init__(self, gcn_in_features, gcn_out_features, bias=True):
        super(MVA, self).__init__()
        # gcn Parameters
        self.gcn_in_features = gcn_in_features
        self.gcn_out_features = gcn_out_features
        # Parameter用于将参数自动加入到参数列表
        self.weight = Parameter(torch.FloatTensor(gcn_in_features, gcn_out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(gcn_out_features))
        else:
            self.register_parameter('bias', None)  # 为模型添加参数
        self.reset_parameters()

        self.fusionsize = 128
        self.max_d = 50
        self.input_dim_drug = 23532
        self.n_layer = 2
        self.emb_size = 384
        self.dropout_rate = 0
        # encoder
        self.hidden_size = 384
        self.intermediate_size = 1536
        self.num_attention_heads = 4 # 2 4 8
        self.attention_probs_dropout_prob = 0.1
        self.hidden_dropout_prob = 0.1

        # specialized embedding with positional one
        self.emb = Embeddings(self.input_dim_drug, self.emb_size, self.max_d, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        # self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
        #                                         self.num_attention_heads, self.attention_probs_dropout_prob,
        #                                         self.hidden_dropout_prob)
        self.fusion = AFF(self.fusionsize)

        # dencoder
        self.decoder_trans_mpnn_cat = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.Dropout(0.1),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 1)
        )
        self.decoder_trans_mpnn_sum = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            # output layer
            nn.Linear(32, 1)
        )

        self.decoder_1 = nn.Sequential(
            nn.Linear(50 * 384, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128)
        )

        self.flatten = nn.Flatten()
        self.decoder_2 = nn.Sequential(
            nn.Linear(600 * 128, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128)
        )

        self.decoder_3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Linear(32, 1)
        )

        self.query_proj = nn.Linear(256, 256 * 2, bias=False)
        self.key_proj = nn.Linear(256, 256 * 2, bias=False)
        self.value_proj = nn.Linear(256, 256 * 2, bias=False)
        self.output_proj = nn.Linear(256 * 2, 256, bias=False)

    def aggregate_message_1(self, nodes, node_neighbours, edges, mask):

        raise NotImplementedError
    def aggregate_message_2(self, nodes, node_neighbours, edges, mask):

        raise NotImplementedError

    # inputs are "batches" of shape (maximum number of nodes in batch, number of features)
    def update_1(self, nodes, messages):
        raise NotImplementedError
    def update_2(self, nodes, messages):
        raise NotImplementedError

    # inputs are "batches" of same shape as the nodes passed to update
    # node_mask is same shape as inputs and is 1 if elements corresponding exists, otherwise 0
    def readout_1(self, hidden_nodes, input_nodes, node_mask):
        raise NotImplementedError
    def readout_2(self, hidden_nodes, input_nodes, node_mask):
        raise NotImplementedError
    def readout(self,input_nodes, node_mask):
        raise NotImplementedError
    def final_layer(self,out):

        raise NotImplementedError

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, fts_1, adjs_1, fts_2, adjs_2, num_size, de_1, de_2):
        num_size = num_size.size(0)

        # GCN encoder
        paddingsize = 600
        # fts_padding_1 = torch.zeros(fts_1.size()[0], paddingsize, 75, dtype=torch.float32).cuda()
        # adjs_padding_1 = torch.zeros(adjs_1.size()[0], paddingsize, paddingsize, dtype=torch.float32).cuda()
        # fts_padding_2 = torch.zeros(fts_2.size()[0], paddingsize, 75, dtype=torch.float32).cuda()
        # adjs_padding_2 = torch.zeros(adjs_2.size()[0], paddingsize, paddingsize, dtype=torch.float32).cuda()
        fts_padding_1 = torch.zeros(fts_1.size()[0], paddingsize, 75, dtype=torch.float32)
        adjs_padding_1 = torch.zeros(adjs_1.size()[0], paddingsize, paddingsize, dtype=torch.float32)
        fts_padding_2 = torch.zeros(fts_2.size()[0], paddingsize, 75, dtype=torch.float32)
        adjs_padding_2 = torch.zeros(adjs_2.size()[0], paddingsize, paddingsize, dtype=torch.float32)
        fts_padding_1[:, :fts_1.size()[1], :] = fts_1
        adjs_padding_1[:, :fts_1.size()[1], :fts_1.size()[1]] = adjs_1
        fts_padding_2[:, :fts_2.size()[1], :] = fts_2
        adjs_padding_2[:, :fts_2.size()[1], :fts_2.size()[1]] = adjs_2

        support_1 = torch.matmul(fts_padding_1, self.weight)
        support_2 = torch.matmul(fts_padding_2, self.weight)
        # print(adjs_padding_1.shape)
        # print(support_1.shape)
        output_1 = torch.matmul(adjs_padding_1, support_1)
        output_2 = torch.matmul(adjs_padding_2, support_2)
        if self.bias is not None:
            output_1 = output_1 + self.bias
            output_2 = output_2 + self.bias

        # Sequence encoder
        ex_d_mask = de_1.unsqueeze(1).unsqueeze(2)
        ex_p_mask = de_2.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0
        # print(de_1.device)
        # print('---------------')
        d_emb = self.emb(de_1)  # num_size x seq_length x embed_size
        # print(de_2.device)
        # print('---------------')
        p_emb = self.emb(de_2)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...

        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())
        p_encoded_layers = self.d_encoder(p_emb.float(), ex_p_mask.float())
        d1_trans_fts = d_encoded_layers.view(num_size, -1)
        d2_trans_fts = p_encoded_layers.view(num_size, -1)

        d1_trans_fts_layer1 = self.decoder_1(d1_trans_fts)
        d2_trans_fts_layer1 = self.decoder_1(d2_trans_fts)
        # feature hybrid
        # d1_cat_fts = d1_trans_fts_layer1 #125,128
        # d2_cat_fts = d2_trans_fts_layer1
        output_1 = self.decoder_2(self.flatten(output_1))
        output_2 = self.decoder_2(self.flatten(output_2))

        output1 = self.fusion(d1_trans_fts_layer1, output_1)
        output2 = self.fusion(d2_trans_fts_layer1, output_2)
        # print(output1.size())
        # x1 = torch.cat((d1_trans_fts_layer1, output_1), dim=1) # 32,256
        #
        # x2 = torch.cat((d2_trans_fts_layer1, output_2), dim=1) # 32,256
        #
        # queries1 = self.query_proj(x1)
        # keys1 = self.key_proj(x1)
        # values1 = self.value_proj(x1)
        # queries2 = self.query_proj(x2)
        # keys2 = self.key_proj(x2)
        # values2 = self.value_proj(x2)
        # batch_size = x1.size(0)
        #
        # queries1 = queries1.view(batch_size, 2, -1)
        # keys1 = keys1.view(batch_size, 2, -1)
        # values1 = values1.view(batch_size, 2, -1)
        #
        # weights1 = torch.matmul(queries1, keys1.transpose(1, 2))
        # weights1 = weights1 / (keys1.size(-1) ** 0.5)
        # weights1 = F.softmax(weights1, dim=-1)
        # print(weights1)
        # attended_values1 = torch.matmul(weights1, values1)
        # attended_values1 = attended_values1.view(batch_size, -1)
        # output1 = self.output_proj(attended_values1)
        #
        # queries2 = queries2.view(batch_size, 2, -1)
        # keys2 = keys2.view(batch_size, 2, -1)
        # values2 = values2.view(batch_size, 2, -1)
        #
        # weights2 = torch.matmul(queries2, keys2.transpose(1, 2))
        # weights2 = weights2 / (keys2.size(-1) ** 0.5)
        # weights2 = F.softmax(weights2, dim=-1)
        #
        # attended_values2 = torch.matmul(weights2, values2)
        # attended_values2 = attended_values2.view(batch_size, -1)
        # output2 = self.output_proj(attended_values2)

        # output1 = torch.cat((d1_trans_fts_layer1, output_1), dim=1)
        # output2 = torch.cat((d2_trans_fts_layer1, output_2), dim=1)
        # # print(output1.shape)
        # print('output1.shape:',output1.shape) # 32,256

        # output1 = d1_trans_fts_layer1 + output_1
        # output2 = d2_trans_fts_layer1 + output_2
        #
        final_fts_cat = torch.cat((output1, output2), dim=1)
        # final_fts_sum = output1 + output2
        #
        result = self.decoder_trans_mpnn_cat(final_fts_cat)
        # result = self.decoder_trans_mpnn_sum(final_fts_sum)
        #
        return result, final_fts_cat

# class AFF(nn.Module):
#     def __init__(self, channels=128, r=4):
#         super(AFF, self).__init__()
#         inter_channels = int(channels // r)
#
#         self.local_att = nn.Sequential(
#             nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.global_att = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(inter_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(channels),
#         )
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x, y):
#         xy = x + y
#         # xy = xy.unsqueeze(0)
#         print(xy.size())
#         xl = self.local_att(xy)
#         xg = self.global_att(xy)
#         xlg = xl + xg
#         wei = self.sigmoid(xlg)
#         xo = x * wei + y * (1 - wei)
#         print(wei)
#         return xo
class AFF(nn.Module):
    def __init__(self, channels=128, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(1, inter_channels, kernel_size=(1, 128), stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 128)),
            nn.Conv2d(1, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        batch_size, feature_size = x.size()

        # Reshape x and y as 2D images
        x = x.view(batch_size, 1, 1, feature_size)
        y = y.view(batch_size, 1, 1, feature_size)

        xy = x + y
        xl = self.local_att(xy)
        xg = self.global_att(xy)
        xlg = xl + xg

        wei = self.sigmoid(xlg.squeeze(dim=2).squeeze(dim=2))
        wei_new = wei.squeeze(dim=1)
        wei_new = torch.mean(wei_new, dim=1, keepdim=True)

        # print(wei_new)
        xo = x.squeeze(dim=2).squeeze(dim=2) * wei + y.squeeze(dim=2).squeeze(dim=2) * (1 - wei)
        xo = xo.squeeze(dim=1)
        return xo

# sub-transformer

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        b = torch.LongTensor(1, 2)
        # b = b.cuda()
        input_ids = input_ids.type_as(b)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)#【1.。。50】

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask


        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)  # +注意力
        attention_output = self.output(self_output, input_tensor)  # +残差
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)  # 给向量加了残差和注意力机制
        intermediate_output = self.intermediate(attention_output)  # 给向量拉长
        layer_output = self.output(intermediate_output, attention_output)  # 把向量带着残差压缩回去

        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states

