import os 
import argparse #解析命令行
import numpy as np 
import random
from scipy.special import softmax
import sys
sys.path.append("./data_loader")#添加自己的搜索目录
sys.path.append("./layer")
sys.path.append("./model")
sys.path.append("./metric")

import torch 
from torch import nn 
from model_config import Config
from mrc_data_loader import MRCNERDataLoader
from mrc_data_processor import *
from optim import AdamW, lr_linear_decay
from bert_mrc import BertQueryNER
from bert_tokenizer import BertTokenizer4Tagger
from mrc_ner_evaluate  import flat_ner_performance, nested_ner_performance
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear




def args_parser():
    parser = argparse.ArgumentParser()
	#argparse是python标准库中调用的命令行解析模块
    #创建 ArgumentParser 对象
    # arguementparser包含将命令行解析成python数据类型所需的全部数据类型

    #通过调用add_argument() 方法给 ArgumentParser对象添加程序所需的参数信息
    parser.add_argument("--config_path", default="configs/zh_bert.json", type=str)
    parser.add_argument("--data_dir", default=None, type=str)
    parser.add_argument("--bert_model", default="data/bert_model/bert-base-chinese-pytorch", type=str)
    parser.add_argument("--task_name", default=None, type=str)
	#在这个demo中提供3个标准数据的预处理函数，如果要使用标准数据集就需要把--task_name命成标准数据集的名字。
    parser.add_argument("--max_seq_length", default=150, type=int)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--dev_batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--checkpoint", default=100, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_proportion", default=-1, type=float)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3006)
    parser.add_argument("--export_model", type=bool, default=True)
    parser.add_argument("--data_sign", type=str, default="msra_ner")
    parser.add_argument("--weight_start", type=float, default=1.0) 
    parser.add_argument("--weight_end", type=float, default=1.0) 
    parser.add_argument("--weight_span", type=float, default=0.0)
    parser.add_argument("--entity_sign", type=str, default="flat")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--entity_threshold", type=float, default=0.0)
    parser.add_argument("--data_cache", type=bool, default=True)

    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--regenerate_rate", type=float, default=0.1)
    parser.add_argument("--STrain", type=int, default=0)
    parser.add_argument("--perepoch", type=int, default=0)

    args = parser.parse_args()
	#通过 parse_args() 方法解析参数

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps #在梯度反传时，每gradient_accumulation_steps次进行一次梯度更新，之前照常利用loss.backward()计算梯度。

    random.seed(args.seed)#当seed()没有参数时，每次生成的随机数是不一样的，而当seed()有参数时，每次生成的随机数是一样的
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)#递归创建output_dir目录
 
    return args


def load_data(config):
#data_loader/mrc_data_processor.py中定义...processor()
    print("-*-"*10)
    print("当前的data_sign是: {}".format(config.data_sign))#命令行中的data_sign输出,比如：ESI阶段的zhwiki、NEE阶段的ecommerce

    if config.data_sign == "conll03":
        data_processor = Conll03Processor()
    elif config.data_sign == "zh_msra":
        data_processor = MSRAProcessor()
    elif config.data_sign == "OntoNotes":
        data_processor = OntoNotesProcessor()
    elif config.data_sign == "en_onto":
        data_processor = Onto5EngProcessor()
    elif config.data_sign == "genia":
        data_processor = GeniaProcessor()
    elif config.data_sign == "ace2004":
        data_processor = ACE2004Processor()
    elif config.data_sign == "ace2005":
        data_processor = ACE2005Processor()
    elif config.data_sign == "resume":
        data_processor = ResumeZhProcessor()
    elif config.data_sign == "wiki":
        data_processor = WikiProcessor()
    elif config.data_sign == "HP":
        data_processor = HPProcessor()
    elif config.data_sign == "HC":
        data_processor = HCProcessor()
    elif config.data_sign == "ecommerce":
        data_processor = EcommerceProcessor()
    elif config.data_sign == "twitter":
        data_processor = TwitterProcessor()
    elif config.data_sign == "project":
        data_processor = ProjectProcessor()
    elif config.data_sign == "baidubaike":
        data_processor = baidubaikeProcessor()    
    else:
        raise ValueError("Please Notice that your data_sign DO NOT exits !!!!!")


    label_list = data_processor.get_labels()#get_labelsdata_loader/定义data_sign.py
    tokenizer = BertTokenizer4Tagger.from_pretrained(config.bert_model, do_lower_case=True)#利用bert生成字向量
    #from_pretrained在layer/bert_basic_model.py
    #这里加载train/dev/test数据在data_loader/mrc_data_loader中定义
    dataset_loaders = MRCNERDataLoader(config, data_processor, label_list, tokenizer, mode="train", allow_impossible=True)
    train_dataloader = dataset_loaders.get_dataloader(data_sign="train") 
    dev_dataloader = dataset_loaders.get_dataloader(data_sign="dev")
    test_dataloader = dataset_loaders.get_dataloader(data_sign="test")
    num_train_steps = dataset_loaders.get_num_train_epochs()


    return train_dataloader, dev_dataloader, test_dataloader, num_train_steps, label_list 

#上边是加载数据，下边是加载模型


def load_model(config, num_train_steps, label_list, pretrain=None):#括号内是函数调用到的参数
    device = torch.device("cuda") #torch.Tensor代表分配到的设备的对象，CPU或者cuda
    n_gpu = config.n_gpu
    model = BertQueryNER(config, )#BertQueryNER模型在model/bert_mrc.py中定义
    if pretrain:
        #数据的长度输出定义在data_loader/mrc_utils.py第280行
        #这里输出的在data_loader/mrc_data_loader.py第80行,data_loader/定义MRCNERDATALoader 45行
        print("等待预训练........")
        pretrained_dict = torch.load(pretrain)#从文件加载用torch.save()保存的对象
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#在字典中，items()函数以列表返回可遍历的(键, 值) 元组数组
        model_dict.update(pretrained_dict)#update更新键值对
        model.load_state_dict(model_dict)#从state_dict 中复制参数和缓冲区到 Module 及其子类中 
        # model.load_state_dict(torch.load(pretrain))
    model.to(device)
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)#用多个gpu运行程序

    param_optimizer = list(model.named_parameters())#named_parameters给出网络层的名字和参数的迭代器


    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
    {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
    {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]

    # optimizer = Adam(optimizer_grouped_parameters, lr=config.learning_rate)
    # 在上面定义的学习率leanrning_rate
    # warmup是一种学习率优化方法（最早出现在ResNet论文中）
	# 在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练
    optimizer = BertAdam(optimizer_grouped_parameters, lr=config.learning_rate, warmup=config.warmup_proportion, t_total=num_train_steps, max_grad_norm=config.clip_grad)

    sheduler = None

    return model, optimizer, sheduler, device, n_gpu


def train(model, optimizer, sheduler,  train_dataloader, dev_dataloader, test_dataloader, config, \
    device, n_gpu, label_list):
    nb_tr_steps = 0 
    tr_loss = 0 

    dev_best_acc = 0 
    dev_best_precision = 0 
    dev_best_recall = 0 
    dev_best_f1 = 0 
    dev_best_loss = 10000000000000


    test_acc_when_dev_best = 0 
    test_pre_when_dev_best = 0 
    test_rec_when_dev_best = 0 
    test_f1_when_dev_best = 0 
    test_loss_when_dev_best = 1000000000000000

    model.train()
    #ESI阶段epochs等于5
    for idx in range(int(config.num_train_epochs)):#idx循环0~5共6次
        tr_loss = 0 
        nb_tr_examples, nb_tr_steps = 0, 0 
        print("#######"*10)
        print("EPOCH: ", str(idx))             #ESI阶段是237208加载，但是打开有237396条数据(因为设置的train_bath_size=16)；#NEE阶段有287314条数据加载，打开也有287314条数据
        print("steps: ", len(train_dataloader))#ESI阶段等于14826 NEE阶段是17958
        #steps等于train_dataloader的长度,train_dataloader的长在139行已经定义过，引用了data_loader/定义MRCNERDATALoader.py77行
        """
        if idx != 0:
            lr_linear_decay(optimizer)
        """
        for step, batch in enumerate(train_dataloader):#enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列,同时列出数据batch和数据下标step
            batch = tuple(t.to(device) for t in batch) # tuple() 函数将列表转换为元组。写入CUDA t.to(device)
            #弄清楚下面这些都是从哪里定义的
            input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate = batch 
            #在这里定义loss  #除了input_ids其他的值都为none
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, \
                start_positions=start_pos, end_positions=end_pos, span_positions=span_pos)
            if n_gpu > 1:
                loss = loss.mean()#.mean()求取平均值，返回的是total_loss
                
            #在模型中loss其实只是在layer/bert_basic_model.py里面调用了交叉熵函数CrossEntropyLoss()
            model.zero_grad()#清空模型中过往梯度
            loss.backward()#反向传播，计算当前梯度
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad) #计算局部范数（快）
            optimizer.step()#根据梯度更新网络参数

            tr_loss += loss.item()#item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回

            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1 


            if nb_tr_steps % config.checkpoint == 0:
                print("-*-"*15)
                print("当前训练(train)的损失为 : ")#输出了4遍，checkpoint一共4000个
                print(loss.item())#正确的损失函数值
                #所有有关于loss的疑问去看model/bert_mrc.py
                #checkpoint检查点：不仅保存模型的参数，优化器参数，还有loss，epoch等（相当于一个保存模型的文件夹）
                """
                 train为训练语料，用于模型训练；
                 dev为开发集，用于模型参数调优；
                 test用于测试
                """
                tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1 = eval_checkpoint(model, dev_dataloader, config, device, n_gpu, label_list, eval_sign="dev")
                print("......"*10)
                print("验证/开发集(dev)的训练结果为: 损失率(loss), 准确率(acc), 精确率(precision), 召回率(recall), F1score(f1)")#验证集阶段
                print(tmp_dev_loss, tmp_dev_acc, tmp_dev_prec, tmp_dev_rec, tmp_dev_f1)#在metric/mrc_ner_evaluate.py中定义这几个取值

                if tmp_dev_f1 > dev_best_f1 :#取tmp_dev_f1最大的
                    dev_best_acc = tmp_dev_acc 
                    dev_best_loss = tmp_dev_loss 
                    dev_best_precision = tmp_dev_prec 
                    dev_best_recall = tmp_dev_rec 
                    dev_best_f1 = tmp_dev_f1 

                    #输出model 
                    if config.export_model:
                        model_to_save = model.module if hasattr(model, "module") else model 
                        output_model_file = os.path.join(config.output_dir, "bert_finetune_model_{}_{}.bin".format(str(idx),str(nb_tr_steps)))#最后生成成的模型名
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("模型保存的地址为:") 
                        print(output_model_file)

                    tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1 = eval_checkpoint(model, test_dataloader, config, device, n_gpu, label_list, eval_sign="test")
                    print("......"*10)
                    print("测试集(test)的训练结果为: 损失率(loss), 准确率(acc), 精确率(precision), 召回率(recall), F1score(f1)")#测试集阶段
                    print(tmp_test_loss, tmp_test_acc, tmp_test_prec, tmp_test_rec, tmp_test_f1)


                    test_acc_when_dev_best = tmp_test_acc 
                    test_pre_when_dev_best = tmp_test_prec
                    test_rec_when_dev_best = tmp_test_rec
                    test_f1_when_dev_best = tmp_test_f1 
                    test_loss_when_dev_best = tmp_test_loss

                print("-*-"*15)



        if config.STrain and idx < (int(config.num_train_epochs) - 1):
            if config.perepoch:
                regenerate = config.regenerate_rate * (1 + idx)
            else:#因为设置的perepoch为0
                regenerate = config.regenerate_rate
                
            print("设置的阈值(regenerate)为:", regenerate)
            train_dataloader = train_regenerate_nospan(model, train_dataloader,
                                            device, config, label_list,
                                            regenerate,
                                            str(idx))

    print("=&="*15)
    print("Best DEV : overall best loss, acc, precision, recall, f1 ")
    print(dev_best_loss, dev_best_acc, dev_best_precision, dev_best_recall, dev_best_f1)
    print("scores on TEST when Best DEV:loss, acc, precision, recall, f1 ")
    print(test_loss_when_dev_best, test_acc_when_dev_best, test_pre_when_dev_best, test_rec_when_dev_best, test_f1_when_dev_best)
    print("=&="*15)
    



def eval_checkpoint(model_object, eval_dataloader, config, \
    device, n_gpu, label_list, eval_sign="dev"):
    #输入数据的类型只能是验证集dev_dataloader或测试集test_dataloader
    model_object.eval()
    #DataParallel说明和前面model一样是由MRCDERQUERY定义的
    #eval() 函数用来执行一个字符串表达式，并返回表达式的值.
    #pytorch会自动把批量归一化和DropOut固定住，不会取每个batchsize的平均，而是用训练好的值。不然的话，有输入数据，即使不训练，它也会改变权值。


    eval_loss = 0 
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    span_gold_lst = []
    end_gold_lst = []
    eval_steps = 0 
    ner_cate_lst = [] 

    for input_ids, input_mask, segment_ids, start_pos, end_pos, span_pos, ner_cate in eval_dataloader:
        input_ids = input_ids.to(device)#.to(device)是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_pos = start_pos.to(device)
        end_pos = end_pos.to(device)
        span_pos = span_pos.to(device) 

        with torch.no_grad():#no_grad输出没有属性
            tmp_eval_loss = model_object(input_ids, segment_ids, input_mask, start_pos, end_pos, span_pos)
            start_logits, end_logits, span_logits = model_object(input_ids, segment_ids, input_mask)#model/bert_mrc.py92行
            start_logits = torch.argmax(start_logits, dim=-1)#返回指定维度(dim=-1)最大值的序号
            end_logits = torch.argmax(end_logits, dim=-1)#同上

        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        span_pos = span_pos.to("cpu").numpy().tolist()

        start_label = start_logits.detach().cpu().numpy().tolist()#反向传播调用到.detach()的时候会停止
        end_label = end_logits.detach().cpu().numpy().tolist()
        span_logits = span_logits.detach().cpu().numpy().tolist()
        span_label = span_logits
        
        input_mask = input_mask.to("cpu").detach().numpy().tolist()

        ner_cate_lst += ner_cate.numpy().tolist()#.tolist转换为列表形式
        eval_loss += tmp_eval_loss.mean().item()#.mean()求取均值；一个元素张量可以用.item()得到元素值
        mask_lst += input_mask 
        eval_steps += 1

        start_pred_lst += start_label 
        end_pred_lst += end_label 
        span_pred_lst += span_label
        
        start_gold_lst += start_pos 
        end_gold_lst += end_pos 
        span_gold_lst += span_pos 

    #这里定义的entity_sign等于flat
    if config.entity_sign == "flat":#metric/mrc_ner_evaluate.py中定义了flat_ner_performance和nested_ner_performance
        eval_accuracy, eval_precision, eval_recall, eval_f1 = flat_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)
    else:
        eval_accuracy, eval_precision, eval_recall, eval_f1 = nested_ner_performance(start_pred_lst, end_pred_lst, span_pred_lst, start_gold_lst, end_gold_lst, span_gold_lst, ner_cate_lst, label_list, threshold=config.entity_threshold, dims=2)
   # print(eval_steps)#32
    average_loss = round(eval_loss / eval_steps, 4)#返回小数点后4位  之后的的四舍五入值。
    eval_f1 = round(eval_f1 , 4)
    eval_precision = round(eval_precision , 4)
    eval_recall = round(eval_recall , 4) 
    eval_accuracy = round(eval_accuracy , 4) 
#调用metric/flat_span_f1.py14行的mask_span_f1函数计算precision，recall，f1.
#调用metric/mrc_ner_evaluate.py145行计算accuracy
#调用model/bert_mrc.py80行计算average_loss
    return average_loss, eval_accuracy, eval_precision, eval_recall, eval_f1

def train_regenerate_nospan(model_object, eval_dataloader, device, config, label_list, gama, saveddata_flag):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    if gama>1:
        gama=config.regenerate_rate*int(1/config.regenerate_rate)#如果gama > 1，gama等于...regenerate_rate就起到定义gama的作用
    model_object.eval()#用来执行一个字符串表达式，并返回表达式的值。
    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_start_pos = []
    train_end_pos = []
    train_ner_cate = []
    start_pred_lst = []
    end_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    end_gold_lst = []
    span_gold_lst= []
    eval_steps = 0
    ner_cate_lst = []
    examples=0

    for input_ids, input_mask, segment_ids, start_true, end_true, span_true, ner_cate in eval_dataloader:
        #examples+=len(input_ids)
        #print("Loading......",str(examples))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_true = start_true.to(device)
        end_true = end_true.to(device)
        span_true =span_true.to(device)

        with torch.no_grad():
            start_logits, end_logits, _ = model_object(input_ids, segment_ids, input_mask)

        start_logits = start_logits.detach().cpu().numpy()#detach()截断反向传播的梯度流
        end_logits = end_logits.detach().cpu().numpy()
        start_true = start_true.to("cpu").numpy()
        end_true = end_true.to("cpu").numpy()
        span_true = span_true.to("cpu").numpy()
        reshape_lst = start_true.shape
#start（end/index）_logits（pos/index）的维度
        start_logits = np.reshape(start_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()
        end_logits = np.reshape(end_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()

        start_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)
        end_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)

        start_index = [[idx for idx, tmp in enumerate(softmax(np.array(j), axis=-1)) if tmp[-1]>gama] for j in start_logits]
        #解码层第一个softmax，idx表示索引，tmp表示概率，只要tmp大于设置的gama（也就是阈值regenerate=0.1）就输出，否则留下
        end_index = [[idx for idx, tmp in enumerate(softmax(np.array(j), axis=-1)) if tmp[-1]>gama] for j in end_logits]
        #同上
        for batch_dim in range(len(start_index)):
            for tmp_start in start_index[batch_dim]:
                tmp_end = [tmp for tmp in end_index[batch_dim] if tmp >= tmp_start]
                if len(tmp_end) == 0:
                    continue
                else:
                    tmp_end = min(tmp_end)
                start_pos[batch_dim][tmp_start]=1
                end_pos[batch_dim][tmp_end]=1


        start_pos = start_pos.tolist()
        end_pos = end_pos.tolist()
        start_true = start_true.tolist()
        end_true = end_true.tolist()
        span_true = span_true.tolist()
        end_pred_lst += end_pos
        start_pred_lst += start_pos
        start_gold_lst += start_true
        end_gold_lst += end_true
        span_gold_lst += span_true
        ner_cate_lst += ner_cate.numpy().tolist()
        mask_lst += input_mask.to("cpu").numpy().tolist()
        start_pos = torch.tensor(start_pos, dtype=torch.short)
        end_pos = torch.tensor(end_pos, dtype=torch.short)
        train_input_ids.append(input_ids)
        train_input_mask.append(input_mask)
        train_segment_ids.append(segment_ids)
        train_start_pos.append(start_pos)
        train_end_pos.append(end_pos)
        train_ner_cate.append(ner_cate)
    train_input_ids = torch.cat(train_input_ids, 0)
    train_input_mask = torch.cat(train_input_mask, 0)
    train_segment_ids = torch.cat(train_segment_ids, 0)
    train_start_pos = torch.cat(train_start_pos, 0)
    train_end_pos = torch.cat(train_end_pos, 0)
    train_ner_cate = torch.cat(train_ner_cate, 0)
    #train_loss_mask = torch.cat(train_loss_mask, 0)

    np.save(config.output_dir + saveddata_flag + "-train_input_ids", train_input_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_input_mask", train_input_mask.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_segment_ids", train_segment_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_start_pos", train_start_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_end_pos", train_end_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_ner_cate", train_ner_cate.cpu().numpy())
    #np.save(config.output_dir + saveddata_flag + "-train_loss_mask", train_loss_mask.cpu().numpy())
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_start_pos, train_end_pos, torch.tensor(np.zeros((train_input_ids.size(0),1,1), dtype=int)), train_ner_cate)
    train_sampler = SequentialSampler(train_data)  # RandomSampler(dataset)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

    print("######Regenerate Over#########")#0.1是表示>0.1的数很多，只有<0.1的被留下，剩下的可以走
    return train_dataloader



def train_regenerate(model_object, eval_dataloader, device, config, label_list, gama, saveddata_flag):
    # input_dataloader type can only be one of dev_dataloader, test_dataloader
    if gama>1:
        gama=config.regenerate_rate*int(1/config.regenerate_rate)
    model_object.eval()
    train_input_ids = []
    train_input_mask = []
    train_segment_ids = []
    train_start_pos = []
    train_end_pos = []
    train_span = []
    train_ner_cate = []
    start_pred_lst = []
    end_pred_lst = []
    span_pred_lst = []
    mask_lst = []
    start_gold_lst = []
    end_gold_lst = []
    span_gold_lst=[]
    eval_steps = 0
    ner_cate_lst = []

    examples=0

    for input_ids, input_mask, segment_ids, start_true, end_true, span_true, ner_cate in eval_dataloader:

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_true = start_true.to(device)
        end_true = end_true.to(device)
        span_true = span_true.to(device)
        ner_cate = ner_cate.to(device)
        with torch.no_grad():
            start_logits, end_logits, span_logits = model_object(input_ids, segment_ids, input_mask)

        start_logits = start_logits.detach().cpu().numpy()
        end_logits = end_logits.detach().cpu().numpy()
        span_logits = span_logits.detach().cpu().numpy()
        start_true = start_true.to("cpu").numpy()
        end_true = end_true.to("cpu").numpy()
        span_true = span_true.to("cpu").numpy()
        reshape_lst = start_true.shape

        start_logits = np.reshape(start_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()
        end_logits = np.reshape(end_logits, (reshape_lst[0], reshape_lst[1], 2)).tolist()
        span_logits = np.reshape(span_logits, (reshape_lst[0], reshape_lst[1], reshape_lst[1], 1)).tolist()

        start_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)
        end_pos = np.zeros([reshape_lst[0], reshape_lst[1]], int)
        span_pred = np.zeros([reshape_lst[0], reshape_lst[1], reshape_lst[1]], int)

        start_index = [[idx for idx, tmp in enumerate(j) if tmp[-1]>gama] for j in softmax(np.array(start_logits))]
        end_index = [[idx for idx, tmp in enumerate(j) if tmp[-1]>gama] for j in softmax(np.array(end_logits))]
        for batch_dim in range(len(start_index)):
            for tmp_start in start_index[batch_dim]:
                tmp_end = [tmp for tmp in end_index[batch_dim] if tmp >= tmp_start]
                if len(tmp_end) == 0:
                    continue
                else:
                    tmp_end = min(tmp_end)
                if span_logits[batch_dim][tmp_start][tmp_end] >= gama:
                    start_pos[batch_dim][tmp_start]=1
                    end_pos[batch_dim][tmp_start]=1
                    span_pred[batch_dim][tmp_start][tmp_end]=1

        start_pos = start_pos.tolist()
        end_pos = end_pos.tolist()
        span_pred=span_pred.tolist()
        start_true = start_true.tolist()
        end_true = end_true.tolist()
        span_true = span_true.tolist()
        end_pred_lst += end_pos
        start_pred_lst += start_pos
        span_pred_lst +=span_pred
        start_gold_lst += start_true
        end_gold_lst += end_true
        span_gold_lst += span_true
        ner_cate_lst += ner_cate.numpy().tolist()
        mask_lst += input_mask.to("cpu").numpy().tolist()
        start_pos = torch.tensor(start_pos, dtype=torch.short)
        end_pos = torch.tensor(end_pos, dtype=torch.short)
        span_pred = torch.tensor(span_pred, dtype=torch.short)
        train_input_ids.append(input_ids)
        train_input_mask.append(input_mask)
        train_segment_ids.append(segment_ids)
        train_start_pos.append(start_pos)
        train_end_pos.append(end_pos)
        train_span.append(span_pred)
        train_ner_cate.append(ner_cate)
    #eval_accuracy, eval_precision, eval_recall, eval_f1 = query_ner_compute_performance(start_pred_lst, end_pred_lst, start_gold_lst, end_gold_lst, ner_cate_lst, label_list, mask_lst, dims=2)
    eval_accuracy, eval_precision, eval_recall, eval_f1 = flat_ner_performance(start_pred_lst, end_pred_lst,
                                                                               span_pred_lst, start_gold_lst,
                                                                               end_gold_lst, span_gold_lst,
                                                                               ner_cate_lst, label_list,
                                                                               threshold=config.entity_threshold,
                                                                               dims=2)

    # eval_accuracy, eval_precision, eval_recall, eval_f1 = compute_performance(pred_lst, gold_lst, mask_lst, label_list, dims=2)

    eval_f1 = round(eval_f1, 4)
    eval_precision = round(eval_precision , 4)
    eval_recall = round(eval_recall , 4)
    eval_accuracy = round(eval_accuracy , 4)
    print("f1: precision:  recall: accuracy")
    print(eval_f1, eval_precision, eval_recall, eval_accuracy)#metric/mrc_ner_evaluate.py
    train_input_ids = torch.cat(train_input_ids, 0)#按照维数0拼接竖着拼
    train_input_mask = torch.cat(train_input_mask, 0)
    train_segment_ids = torch.cat(train_segment_ids, 0)
    train_start_pos = torch.cat(train_start_pos, 0)
    train_end_pos = torch.cat(train_end_pos, 0)
    train_span = torch.cat(train_span, 0)
    train_ner_cate = torch.cat(train_ner_cate, 0)
    #train_loss_mask = torch.cat(train_loss_mask, 0)

    np.save(config.output_dir + saveddata_flag + "-train_input_ids", train_input_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_input_mask", train_input_mask.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_segment_ids", train_segment_ids.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_start_pos", train_start_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_end_pos", train_end_pos.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_span", train_span.cpu().numpy())
    np.save(config.output_dir + saveddata_flag + "-train_ner_cate", train_ner_cate.cpu().numpy())
    #np.save(config.output_dir + saveddata_flag + "-train_loss_mask", train_loss_mask.cpu().numpy())
    train_data = TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_start_pos, train_end_pos, train_span, train_ner_cate)#类似于python的zip
    train_sampler = SequentialSampler(train_data)  # 顺序采样数据以及随机采样数据：RandomSampler(dataset)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=config.train_batch_size)

    print("######Regenerate Over#########")
    return train_dataloader



def merge_config(args_config):
    model_config_path = args_config.config_path 
    model_config = Config.from_json_file(model_config_path)
    model_config.update_args(args_config)
    model_config.print_config()
    return model_config



def main():
    args_config = args_parser()
    config = merge_config(args_config)
    train_loader, dev_loader, test_loader, num_train_steps, label_list = load_data(config)
    model, optimizer, sheduler, device, n_gpu = load_model(config, num_train_steps, label_list, config.pretrain)
    train(model, optimizer, sheduler, train_loader, dev_loader, test_loader, config, device, n_gpu, label_list)
    

if __name__ == "__main__":
    main() 
