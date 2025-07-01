args_path=/home/slee/BatteryLife/checkpoints/CPTransformer_sl1_lr5e-05_dm128_nh4_el12_dl0_df256_lradjconstant_datasetTongji_lossMSE_wd0.0_wlFalse_bs16_s2021-CPTransformer/ # the model you want to evaluate
batch_size=16
num_process=2
master_port=26949
eval_cycle_min=1 # set eval_cycle_min as 1 and eval_cycle_max as 100 to evaluate all samples
eval_cycle_max=100
eval_dataset=Tongji
model=CPTransformer

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes $num_process --main_process_port $master_port evaluate_model.py \
  --args_path $args_path \
  --batch_size $batch_size \
  --eval_cycle_min $eval_cycle_min \
  --eval_cycle_max $eval_cycle_max \
  --eval_dataset $eval_dataset \
  --model $model