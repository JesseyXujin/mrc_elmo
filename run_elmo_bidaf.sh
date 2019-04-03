export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_PATH="/home/ssd3/tianxin04/DuReader/data/preprocessed/"

python run.py   \
--train \
--elmo \
--pretrain_elmo_model_path "./pretrain_models/1228000/" \
--elmo_dict_dir "/home/ssd3/tianxin04/data/baike_mini/vocabulary_min5k.txt" \
--trainset "${DATA_PATH}/trainset/search.train.json" \
          "${DATA_PATH}/trainset/zhidao.train.json" \
--devset "${DATA_PATH}/devset/search.dev.json" \
         "${DATA_PATH}/devset/zhidao.dev.json" \
--testset "${DATA_PATH}/testset/search.test.json" \
          "${DATA_PATH}/testset/zhidao.test.json" \
--vocab_dir "/home/ssd3/tianxin04/DuReader/data/vocab" \
--use_gpu true \
--save_dir ./models \
--pass_num 5 \
--learning_rate 0.001 \
--batch_size 8 \
--embed_size 300 \
--hidden_size 150 \
--max_p_num 5 \
--max_p_len 500 \
--dev_interval 1000 \
--max_q_len 60 \
--max_a_len 200 \
--weight_decay 0.0 \
--elmo_l2_coef 0.001 \
--log_interval 50 \
--drop_rate 0.2 $@
