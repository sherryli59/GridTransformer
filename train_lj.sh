USE_IDA="${USE_IDA:-true}"
USE_IDA_FLAG="--use_ida"
if [[ "${USE_IDA,,}" == "false" || "${USE_IDA}" == "0" ]]; then
  USE_IDA_FLAG="--no-use_ida"
fi

python train.py \
  --dataset lj_transferable \
  --data_dir /mnt/ssd/mcmc/lj_N16_T1.h5 \
  --lj_transfer_hilbert_resolution 128 \
  --lj_transfer_window 3.0 \
  --lj_transfer_bins 64 \
  --ar_max_dist 4.0 \
  ${USE_IDA_FLAG} \
  --epochs 10 \
  --ckpt_dir lj_ckpts_transfer \

python sample_lj.py \
  --mode relative \
  --ckpt lj_ckpts_transfer/best.ckpt \
  --Lx 4 --Ly 4 \
  --num_particles 16 \
  ${USE_IDA_FLAG} \
  --relative_window 3.0 \
  --relative_bins 64 \
  --nsamples 100000 \
  --sample_batch_size 512 \
  --save lj_ckpts_transfer/samples.npz
