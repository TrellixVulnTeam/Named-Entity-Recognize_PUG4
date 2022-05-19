
CUDA_VISIBLE_DEVICES=10 python -u run/train_bert_mrc.py --data_dir data/train_for_ESI/baidubaike/ --checkpoint 20000 --learning_rate 6e-6 --num_train_epochs 5 --output_dir data/saved_model/wiki/ --data_sign wiki

CUDA_VISIBLE_DEVICES=15 python -u run/train_bert_mrc.py --data_dir data/train_for_NEE/ecommerce --checkpoint 4000 --learning_rate 3e-5 --num_train_epochs 6 --output_dir data/saved_model/ecommerce/stage2 --data_sign ecommerce --pretrain data/saved_model/zhwiki/bert_finetune_model.bin --bert_model data/bert_model/bert-base-chinese-pytorch/ --warmup_proportion 0.4 --regenerate_rate 0.1 --STrain 1 --perepoch 0

CUDA_VISIBLE_DEVICES=10  python -u run/train_cluster_bert_mrc.py --data_dir data/train_for_FET/ecommerce/ --checkpoint 2000 --learning_rate 2e-5 --num_train_epochs 5 --output_dir data/saved_model/ecommerce/stage3 --data_sign ecommerce --pretrain data/saved_model/ecommerce/stage2/ --bert_model data/bert_model/bert-base-chinese-pytorch/ --num_clusters 23 --gama 0.001 --clus_niter 60 --dropout_rate 0.1


CUDA_VISIBLE_DEVICES=11 python -u run/train_bert_mrc.py --data_dir data/train_for_NEE/OntoNotes --checkpoint 4000 --learning_rate 3e-5 --num_train_epochs 6 --output_dir data/saved_model/OntoNotes/stage2 --data_sign OntoNotes --pretrain data/saved_model/zhwiki/bert_finetune_model.bin --bert_model data/bert_model/bert-base-chinese-pytorch/ --warmup_proportion 0.4 --regenerate_rate 0.1 --STrain 1 --perepoch 0

CUDA_VISIBLE_DEVICES=11  python -u run/train_cluster_bert_mrc.py --data_dir data/train_for_FET/ontonotes/ --checkpoint 2000 --learning_rate 2e-5 --num_train_epochs 5 --output_dir data/saved_model/OntoNotes/stage3 --data_sign OntoNotes --pretrain data/saved_model/OntoNotes/stage2/ --bert_model data/bert_model/bert-base-chinese-pytorch/ --num_clusters 23 --gama 0.001 --clus_niter 60 --dropout_rate 0.1
