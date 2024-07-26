# Setup 
## Environment 
Modules that were active in my environment: 

1) dot                
2) rhel8/global     
3) rhel8/slurm                    
4) rhel8/cclake-env/2024-06-01            
5) rhel8/cclake/base                 
6) glibc/2.28/gcc/4d2cj5pu                       
7) gcc-runtime/13.3.0/gcc/2sn7kkm3         
8) zlib-ng/2.1.6/gcc/mkjcey6m          
9) zstd/1.5.6/gcc/l5c7mfsx                         
10) binutils/2.42/gcc/y33xj6rq                              
11) intel-oneapi-compilers/2023.2.4/gcc/4lbvg4hv           
12) intel-oneapi-mpi/2021.12.1/intel/hlruk24n              
13) rhel8/cclake/intel                          
14) rhel8/default-ccl                                      
15) singularity/current                       
16) cuda/11.4
17) libpciaccess/0.16/gcc-9.4.0-6fonbj6
18) libiconv/1.16/gcc-9.4.0-ahebbov
19) libxml2/2.9.12/gcc-9.4.0-gnknt5e
20)  ncurses/6.2/gcc-9.4.0-aiirok7
21)  hwloc/2.5.0/gcc-9.4.0-7sqomga
22)  libevent/2.1.12/gcc-9.4.0-hgny7cm
23)  numactl/2.0.14/gcc-9.4.0-52dwc6n
24)  cuda/11.4.0/gcc-9.4.0-3hnxhjt
25)  gdrcopy/2.2/gcc-9.4.0-e4igtfp
26)  knem/1.1.4/gcc-9.4.0-bpbxgva
27)  libnl/3.3.0/gcc-9.4.0-whwhrwb
28)  rdma-core/34.0/gcc-9.4.0-5eo5n2u
29)  ucx/1.11.1/gcc-9.4.0-lktqyl4
30)  openmpi/4.1.1/gcc-9.4.0-epagguv(default)
31)  rhel8/default-gpu

For packages, run: 
```
pip install -r requirements.txt
```
## Access
Go to https://huggingface.co/meta-llama/Llama-2-7b-hf and request access to the llama-2 7b model

Go to settings and create an access token for the llama-2 7b model. 
Go to src/utils/others.py and insert your access token on line 42 and 86 

## Download Minipile 
To install the minipile dataset, run 
```
git clone https://huggingface.co/datasets/JeanKaddour/minipile
```
Or go to https://huggingface.co/datasets/JeanKaddour/minipile to find alternative ways 

## Data Construction 
1. Split long text:
```
python3 ./src/utils/split_long_text.py \
  --base_model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --blending_model_name_or_path "openlm-research/open_llama_7b_v2" \
  --another_blending_model_name_or_path "mosaicml/mpt-7b" \
  --dataset "<path_to_minipile>" \
  --dataset_save_dir "<path_to_minipile_split>" \
  --cache_dir "<path_to_cache_dir>" \
  --block_size 2048 \
  --preprocessing_num_workers 80
```

2. Get representations for each LLM (Do this step 3x. Once for Llama-2, once for Open Llama and once for MPT)

```
for i in {0..7}; do
export CUDA_VISIBLE_DEVICES=${i}
python ./src/utils/forward_for_logits.py \
  --model_name_or_path "<path_to_each_model>" \
  --dataset "<path_to_minipile_split>" \
  --dataset_save_dir "${i}_8_<path_to_minipile_split_each_model_representation>" \
  --dataset_split_num 8 \
  --dataset_index ${i} \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --training_mode full \
  --load_in_half bf16 \
  --batch_size 8 \
  --preprocessing_num_workers 80 \
  --top_k_logits 10 \
  --save_per_token_metric 2>&1 > "${i}_8_<path_to_log_file>" 2>&1 &
unset CUDA_VISIBLE_DEVICES
sleep 30
done

wait
```

3. Align representations from different LLMs
```
# Get vocab mapping from different LLMs.

# llama_2_7b <-> open_llama_7b_v2
python ./src/utils/vocab_mapping.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_open_llama_7b_v2>" \
  --dataset_dir "<path_to_minipile_split>" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_open_llama_7b_v2_vocab_mapping>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --vocab_mapping_type "default" \
  --num_process 1

# llama_2_7b <-> mpt_7b
python ./src/utils/vocab_mapping.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_mpt_7b>" \
  --dataset_dir "<path_to_minipile_split>" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_mpt_7b_vocab_mapping>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --vocab_mapping_type "default" \
  --num_process 1
```
```
# Align representations from different LLMs.

# llama_2_7b <-> open_llama_7b_v2
for i in {0..7}; do
python ./src/utils/token_alignment.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_open_llama_7b_v2>" \
  --base_dataset_dir "${i}_8_<path_to_minipile_split_llama_2_7b_representation>" \
  --blending_dataset_dir "${i}_8_<path_to_minipile_split_open_llama_7b_v2_representation>" \
  --dataset_save_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_aligned_representation>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --preprocessing_num_workers 80 \
  --batch_size 100 \
  --blending_model_index 0 \
  --vocab_align_type "soft" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_open_llama_7b_v2_vocab_mapping>" \
  --metric_level "sequence"
done 

# llama_2_7b <-> mpt_7b
for i in {0..7}; do
python ./src/utils/token_alignment.py \
  --base_model_name_or_path "<path_to_llama_2_7b>" \
  --blending_model_name_or_path "<path_to_mpt_7b>" \
  --base_dataset_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_aligned_representation>" \
  --blending_dataset_dir "${i}_8_<path_to_minipile_split_mpt_7b_representation>" \
  --dataset_save_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_mpt_7b_aligned_representation>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --preprocessing_num_workers 80 \
  --batch_size 100 \
  --blending_model_index 1 \
  --vocab_align_type "soft" \
  --vocab_mapping_save_dir "<path_to_llama_2_7b_mpt_7b_vocab_mapping>" \
  --metric_level "sequence"
done
```
4. Packing all features to speed up training.
```
for i in {0..7}; do
python3 ./src/utils/packing.py \
  --dataset_dir "${i}_8_<path_to_minipile_split_llama_2_7b_open_llama_7b_v2_mpt_7b_aligned_representation>" \
  --dataset_save_dir "${i}_8_<path_to_miniplie_fusellm_processed>" \
  --cache_dir "<path_to_cache_dir>" \
  --model_max_length 2048 \
  --preprocessing_num_workers 80 \
  --batch_size 1000 \
  --metric_level "sequence"
```
## Training 
Training code instructions for FuseLLM: 
```
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

deepspeed --master_port=20001 ./src/train.py \
  --training_mode full \
  --deepspeed ./config/zero_stage2_config.json \
  --model_name_or_path "<path_to_llama_2_7b>" \
  --output_dir "<path_to_save_fusellm_7b>" \
  --model_max_length 2048 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 500 \
  --save_total_limit 1 \
  --evaluation_strategy steps \
  --per_device_eval_batch_size 1 \
  --logging_strategy steps \
  --do_train \
  --do_eval \
  --bf16 True \
  --tf32 True \
  --warmup_ratio 0.008 \
  --lr_scheduler_type cosine \
  --dataset_name "0_8_<path_to_miniplie_fusellm_processed>,1_8_<path_to_miniplie_fusellm_processed>,2_8_<path_to_miniplie_fusellm_processed>,3_8_<path_to_miniplie_fusellm_processed>,4_8_<path_to_miniplie_fusellm_processed>,5_8_<path_to_miniplie_fusellm_processed>,6_8_<path_to_miniplie_fusellm_processed>,7_8_<path_to_miniplie_fusellm_processed>" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --eval_steps 500 \
  --optim adamw_torch \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --learning_rate 1e-5 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --seed 42 \
  --gradient_checkpointing True \
  --use_flash_attn True \
  --report_to tensorboard \
  --do_distill \
  --distill_with_ref_model True \
  --distill_with_aligned_model_0 True \
  --distill_with_aligned_model_1 True \
  --distill_loss_type "ce" \
  --distill_teacher_temperature 1.0 \
  --lm_loss_weight 0.9 \
  --distill_greater_as_gt True \
  --distill_greater_as_gt_type "hard" \
  --dataloader_num_workers 10 \
  --remove_unused_columns False 2>&1 | tee "<path_to_log_file>"
```
## Evaluation 
Go to https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh and download all BBH tasks. 
Then run: 
```
python3 run_eval.py
   --data_dir <path to bbh data>
   --save_dir <path to save dir>
   --model_name_or_path <path to FuseLLM>
```
