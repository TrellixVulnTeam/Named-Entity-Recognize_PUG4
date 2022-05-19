import os 
import sys 
import copy 
import json 
import math 
import logging 
import numpy as np



root_path = "/".join(os.path.realpath(__file__).split("/")[:-2])
print("the root_path of current file is: ")
print(root_path)
if root_path not in sys.path:
    sys.path.insert(0, root_path)


import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


from layer.bert_basic_model import BertModel, PreTrainedBertModel, BertConfig


class BertMRCNER(nn.Module):
    """
    问答的BERT模型(span_extraction)该模块由顶部为线性的BERT模型组成计算start_logits和end_logits的序列输出。
    一个带有配置的BertConfig类实例来构建一个新模型。
    输入：
        input_ids: torch.LongTensor. of shape [batch_size, sequence_length]
        token_type_ids: an optional torch.LongTensor, [batch_size, sequence_length]
            of the token type [0, 1]. Type 0 corresponds to sentence A, Type 1 corresponds to sentence B.
        attention_mask: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with index select [0, 1]. it is a mask to be used if the input sequence length is smaller
            than the max input sequence length in the current batch.
        start_positions: positions of the first token for the labeled span. torch.LongTensor
            of shape [batch_size, seq_len], if current position is start of entity, the value equals to 1.
            else the value equals to 0.
        end_position: position to the last token for the labeled span.
            torch.LongTensor, [batch_size, seq_len]
    输出:total_loss,如果"start_positions" 和 "end_positions"不是空的
    """

    def __init__(self, config):
        super(BertMRCNER, self).__init__()
        bert_config = BertConfig.from_dict(config.bert_config.to_dict())
        self.bert = BertModel(bert_config)

        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        self.hidden_size = config.hidden_size
        self.bert = self.bert.from_pretrained(config.bert_model)
        self.cluster_layer = config.cluster_layer

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        sequence_output, _, _, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                             output_all_encoded_layers=False)
        sequence_output = sequence_output.view(-1, self.hidden_size)

        start_logits = self.start_outputs(sequence_output)
        end_logits = self.end_outputs(sequence_output)

        if start_positions is not None and end_positions is not None:
            loss_fct = CrossEntropyLoss()

            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            # total_loss = start_loss + end_loss + span_loss
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

class BertMRCNER_CLUSTER(nn.Module):
    def __init__(self, config):
        super(BertMRCNER_CLUSTER, self).__init__()
        bert_config = BertConfig.from_dict(config.bert_config.to_dict()) 
        self.bert = BertModel(bert_config)

        self.start_outputs = nn.Linear(config.hidden_size, 2) #nn.linear设置了网络中的全连接层(维度)
        self.end_outputs = nn.Linear(config.hidden_size, 2)#hidden_size是隐层节点神经元

        self.cluster_classify = nn.Linear(config.hidden_size, config.num_clusters)#（768，23）维度

        self.hidden_size = config.hidden_size 
        self.bert = self.bert.from_pretrained(config.bert_model)

        self.margin = config.margin

        self.gama = config.gama
        self.cluster_layer = config.cluster_layer
        self.pool_mode = config.pool_mode#sum

        self.drop=nn.Dropout(config.dropout_rate)

    def KLloss(self, probs1, probs2):
        loss = nn.KLDivLoss()
        log_probs1 = F.log_softmax(probs1, 1)
        probs2 = F.softmax(probs2, 1)
        return loss(log_probs1, probs2)


    def get_features(self, input_ids, token_type_ids=None, attention_mask=None,
                     start_positions=None, end_positions=None):
        sequence_output, _, _ = self.bert(input_ids, token_type_ids, attention_mask,
                                             output_all_encoded_layers=False)
        sequence_output = sequence_output.view(-1, self.hidden_size)
        start_positions = start_positions.view(-1)
        end_positions = end_positions.view(-1)

        start_pos = np.argwhere(start_positions.cpu().numpy()==1)
        end_pos = np.argwhere(end_positions.cpu().numpy()==1)

        start_pos = np.reshape(start_pos, (len(start_pos))).tolist()
        end_pos = np.reshape(end_pos, (len(end_pos))).tolist()
        features=[]
        for i, s in enumerate(start_pos):
            if i >=len(end_pos):
                continue
            e = end_pos[i]
            if len(features)==0:
                features = sequence_output[s:e+1]
                if self.pool_mode == "sum":#走这里，对输入的tensor数据的维度等于0（行）求和
                    features = torch.sum(features, dim=0, keepdim=True)
                elif self.pool_mode == "avg":
                    features = torch.mean(features, dim=0, keepdim=True)
                elif self.pool_mode=="max":
                    features = features.transpose(0, 1).unsqueeze(0)
                    features = F.max_pool1d(input=features, kernel_size=features.size(2)).transpose(1, 2).squeeze(0)
            else:
                aux = sequence_output[s:e+1]
                if self.pool_mode == "sum":
                    aux = torch.sum(aux, dim=0, keepdim=True)
                elif self.pool_mode == "avg":
                    aux = torch.mean(aux, dim=0, keepdim=True)
                elif self.pool_mode == "max":
                    aux = aux.transpose(0, 1).unsqueeze(0)
                    aux = F.max_pool1d(input=aux, kernel_size=aux.size(2)).transpose(1, 2).squeeze(0)
                features = torch.cat((features, aux), 0)
        return features

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None, span_positions=None, input_truth=None,
                cluster_var=None):
        sequence_output, _, _ = self.bert(input_ids, token_type_ids, attention_mask,#sequence_out预测单词的输入值
                                             output_all_encoded_layers=False)
        
        start_logits = self.start_outputs(sequence_output)#变全连接神经网络 的 预测单词输入值
        end_logits = self.end_outputs(sequence_output)

        sequence_output = sequence_output.view(-1, self.hidden_size)#把sequence_output变换为hidden_size列，但是不知道变为几行

        if start_positions is not None and end_positions is not None:#因为start/end_position非空所以走这里
            loss_fct = CrossEntropyLoss()#交叉熵损失函数

            start_positions = start_positions.view(-1).long()#不确定想要换成什么维度的时候用-1
            end_positions = end_positions.view(-1).long()

            #ner_loss
            #start_loss是把startLogits(不确定想要几行，但确定想要2列)和start_position进行交叉熵操作
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions)#start_loss=预测的实体开始的位置与真实开始位置的交叉熵损失
            #同上
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions)#end_loss=预测的实体的结束位置 与 真实结束位置的交叉熵损失
            #total_loss = start_loss + end_loss + span_loss
            total_loss = (start_loss + end_loss) / 2 #求取平均值

            if input_truth is not None:#loss走这里
                #聚类的损失
                loss_fct_cluster = CrossEntropyLoss(cluster_var)
                start_pos = np.argwhere(start_positions.cpu().numpy()==1)
                end_pos = np.argwhere(end_positions.cpu().numpy()==1)
                start_pos = np.reshape(start_pos, (len(start_pos))).tolist()
                end_pos = np.reshape(end_pos, (len(end_pos))).tolist()
                features=[]
                for i, s in enumerate(start_pos):
                    if i >=len(end_pos):
                        continue
                    e = end_pos[i]
                    if i==0:
                        features = sequence_output[s:e + 1]#从实体开始位置到结束位置
                        if self.pool_mode == "sum":#pool_mode是"sum"
                            features = torch.sum(features, dim=0, keepdim=True)#降维，纵向压缩，就是说每列相加，压缩成一行
                        elif self.pool_mode == "avg":
                            features = torch.mean(features, dim=0, keepdim=True)
                        elif self.pool_mode == "max":
                            features = features.transpose(0, 1).unsqueeze(0)
                            features = F.max_pool1d(input=features, kernel_size=features.size(2)).transpose(1, 2).squeeze(0)
                    else:

                        aux = sequence_output[s:e + 1]
                        if self.pool_mode == "sum":
                            aux = torch.sum(aux, dim=0, keepdim=True)
                        elif self.pool_mode == "avg":
                            aux = torch.mean(aux, dim=0, keepdim=True)
                        elif self.pool_mode == "max":
                            aux = aux.transpose(0, 1).unsqueeze(0)
                            aux = F.max_pool1d(input=aux, kernel_size=aux.size(2)).transpose(1, 2).squeeze(0)
                        features = torch.cat((features, aux), 0)


                if len(features)==0:
                    return total_loss
                features=self.drop(features)#drop_rate=0.1 dropout解决模型过拟合问题
                prob = self.cluster_classify(features)
                #print("计算prob....................")#一个循环这个循环随着训练模型得出loss等一起进行
                #print(prob)
                """
                
tensor([[-4.6483e-01,  9.6603e-01, -5.1883e-01, -1.6455e-01, -1.8465e-01,
         -4.3050e-01,  4.8578e-01, -2.0032e+00, -1.0221e+00, -1.7868e+00,
         -1.2755e+00, -6.7367e-01, -9.8275e-01, -5.2716e-01, -1.5022e-01,
          1.7702e+00,  2.7048e+00, -4.5436e-01,  1.4431e+00, -4.3316e-01,
         -4.6120e-01,  1.0386e+00, -1.2383e+00],
        [ 6.2606e-01,  3.4211e-01,  1.0379e+00, -1.7616e+00, -1.0756e+00,
         -6.7220e-02,  1.7219e+00, -1.0153e+00,  9.4916e-01, -1.4764e+00,
         -1.6028e+00, -9.8628e-01, -2.7046e-01, -1.0255e+00, -1.3429e+00,
          2.5591e+00,  3.3242e+00, -1.2443e+00,  1.7937e+00, -1.0627e+00,
         -7.2473e-01,  2.4697e+00, -1.7014e+00],
        [ 1.6292e-01, -3.7197e-01,  5.0518e-01, -3.8805e-01, -6.1459e-01,
         -4.6643e-01, -7.1699e-01, -1.3532e+00, -1.2904e+00, -4.8059e-02,
         -1.2238e-01,  3.0380e-01,  1.5654e-02,  6.6365e-01, -1.2919e+00,
         -7.2568e-01, -1.0645e-01, -5.9499e-01,  3.9081e-01,  7.3424e-02,
          5.1159e-01,  1.1715e+00,  8.2101e-01],
        [-1.2384e+00,  7.0013e-01, -4.2738e-01, -5.7879e-01, -6.9111e-01,
         -5.8124e-01, -8.5089e-01, -1.1191e+00,  8.3528e-01, -1.0079e+00,
         -1.5663e+00, -1.9075e-01, -2.7413e-01, -6.3493e-01, -7.7683e-02,
          5.9780e+00,  3.4829e+00, -1.1006e+00, -3.6623e-01, -1.4053e+00,
         -9.6609e-01,  1.8866e+00, -8.1971e-01],
        [ 1.2541e+00,  8.0078e-01,  6.5151e-01, -1.0092e+00,  3.4681e-01,
          7.9692e-01,  5.9284e-01, -1.5488e-01, -1.2451e+00,  8.2403e-01,
         -1.2775e+00, -3.9277e-01, -1.4320e-01, -2.4424e-01, -2.0594e+00,
         -1.6829e+00, -1.3145e+00, -1.1632e+00,  1.3306e+00,  4.8760e-01,
         -9.9729e-01,  4.6096e-01,  6.3059e-01],
        [-2.4567e-01,  7.4564e-01,  2.9987e-01,  1.1950e-01,  4.9181e-02,
         -8.3450e-01, -7.4274e-01, -1.3871e+00, -1.4511e+00,  3.0542e-01,
         -2.3727e-01, -4.7368e-01, -3.9393e-01,  7.0563e-01, -1.4528e+00,
         -1.9415e+00, -8.2039e-01, -3.4968e-01,  4.0623e-01, -1.5474e-02,
          4.0440e-01,  7.4706e-01,  1.2052e+00],
        [-2.6809e-01,  4.2366e-03,  4.2255e-01, -6.5519e-01, -1.3351e+00,
          1.6950e+00, -9.5995e-01, -1.3189e+00, -6.7324e-01,  4.5940e-01,
         -9.8930e-01,  1.8602e+00, -1.2334e+00,  1.7049e-01, -1.7413e+00,
         -1.8678e+00, -1.8928e+00, -1.2870e-01,  6.3532e-01, -3.0367e-01,
          2.1497e-01,  1.9997e+00,  2.7250e-01],
        [-5.9387e-02, -9.8745e-01, -2.6617e-01, -2.9616e-01, -1.5940e+00,
          1.9305e+00, -1.4699e+00, -2.3857e+00, -2.5160e+00,  6.2527e-01,
         -4.2580e-01,  1.4058e+00, -9.5031e-01,  5.6403e-01, -1.9942e+00,
         -3.1039e+00, -3.2782e+00, -1.0215e+00,  2.1469e-01, -1.1827e+00,
          7.9897e-01,  2.7141e+00,  1.2836e+00]], device='cuda:0',
       grad_fn=<AddmmBackward>)
                """
                CEloss1=loss_fct_cluster(prob, input_truth[:len(prob)])#特征分到聚类的损失与真实列表
                #CEloss2=loss_fct(prob_C, input_truth[:len(prob_C)])
                #KL=self.KLloss(prob, prob_C)
                #cluster_loss=CEloss1+CEloss2+KL

                #cluster_loss = loss_fct_cluster(cluster, input_truth[:len(cluster)])
                #print("total_loss:  ",total_loss)
                #print("cluster_loss:    ", cluster_loss)
                return total_loss + self.gama*CEloss1 #config中设gama=0.9
            else:
                return total_loss #eval_loss走这儿了
        else:

            span_logits = torch.ones(start_logits.size(0), start_logits.size(1), start_logits.size(1)).cuda()
            return start_logits, end_logits, span_logits
