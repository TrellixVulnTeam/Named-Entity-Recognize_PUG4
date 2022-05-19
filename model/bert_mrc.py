# Bert Model for MRC-Based NER Task


import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss 
import sys
sys.path.append("./layer")

from bert_basic_model import BertModel, BertConfig


class BertQueryNER(nn.Module):
    def __init__(self, config):
        super(BertQueryNER, self).__init__()
        bert_config = BertConfig.from_dict(config.bert_config.to_dict()) 
        self.bert = BertModel(bert_config)
        #nn.linear是用于设置网络中的全连接层的，输入输出都得是二维张量
        self.start_outputs = nn.Linear(config.hidden_size, 2) #2代表输入的二维张量的形状 输出层
        self.end_outputs = nn.Linear(config.hidden_size, 2) #输出张量的形状  输出层

        #self.span_embedding = MultiNonLinearClassifier(config.hidden_size*2, 1, config.dropout)
        self.hidden_size = config.hidden_size #configs中的.json配置
        self.bert = self.bert.from_pretrained(config.bert_model) 
        self.loss_wb = config.weight_start #1
        self.loss_we = config.weight_end #1
        self.loss_ws = 0

#前向传播,想要的请看layer/bert_basic_model.py
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, 
        start_positions=None, end_positions=None, span_positions=None):
        """
            start_positions: (batch x max_len x 1)  开始位置
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
            end_positions: (batch x max_len x 1)  结束位置
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
            span_positions: (batch x max_len x max_len) 跨度长度
                
        """

        sequence_output, pooled_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        sequence_heatmap = sequence_output # batch x seq_len x hidden
        batch_size, seq_len, hid_size = sequence_heatmap.size()#size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数。

        start_logits = self.start_outputs(sequence_heatmap) # batch x seq_len x 2 o

        end_logits = self.end_outputs(sequence_heatmap) # batch x seq_len x 2
        if self.loss_ws:#只要不是0就运行 #但是这里在前面定义了是0
            print("You are computing span loss")
         
            start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)#在第二个维度上增加一个维度
            end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)#在第一维新增加一个维度
            #start_end_concat[0]的形状是 : batch x 1 x seq_len x 2*hidden

            span_matrix = torch.cat([start_extend, end_extend], 3) # batch x seq_len x seq_len x 2*hidden把两个按照维数3拼接起来

            span_logits = self.span_embedding(span_matrix)  # batch x seq_len x seq_len x 1
            span_logits = torch.squeeze(span_logits)  # batch x seq_len x seq_len

        if start_positions is not None and end_positions is not None:#输入的数据有start/end_position loss是start_logits的CrossEntropy与end_logits的CrossEntropy的均值。
            loss_fct = CrossEntropyLoss() #预测概率分布与真实概率分布之间的交叉熵损失函数      #long() 函数将数字或字符串转换为一个长整型。python3X已经删除该函数
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1).long())#.view表示拼接成什么形状，这里-1表示不知道几行，但是确定想要两列
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1).long())
            if self.loss_ws:#因loss_ws是0，所以不运行这里
                span_loss_fct = nn.BCEWithLogitsLoss()# https://blog.csdn.net/qq_22210253/article/details/85222093
                span_loss = span_loss_fct(span_logits.view(batch_size, -1), span_positions.view(batch_size, -1).float())
                total_loss = self.loss_wb * start_loss + self.loss_we * end_loss + self.loss_ws * span_loss   
            else:#因为loss_ws是0，就运行这里，得到tmp_eval_loss
                total_loss = self.loss_wb * start_loss + self.loss_we * end_loss # total = start + end
            return total_loss 
            """
            ！看上面！数据集里面规定了start_position 和 end_position
            """
        else:                             
            if self.loss_ws:
                span_logits = torch.sigmoid(span_logits) # batch x seq_len x seq_len

            else:# start_logits, end_logits, span_logits 在这里得到
                span_logits = torch.ones(start_logits.size(0), start_logits.size(1), start_logits.size(1)).cuda()#返回一个全为1 的张量，形状由可变参数size定义 .size(,axis=0)返回行数/.size(,axis=1)返回列数
            return start_logits, end_logits, span_logits








