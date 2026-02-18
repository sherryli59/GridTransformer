python train.py --dataset lj_cell --data_dir /mnt/ssd/mcmc/lj_N16_T1.h5 --lj_cell_resolution 64 --epochs 10 --ckpt_dir lj_ckpts
python sample_lj.py --ckpt lj_ckpts/best.ckpt --Lx 4 --Ly 4 --num_particles 16 --nsamples 100000 --sample_batch_size 512 --save lj_ckpts/samples.npz
python sample_lj.py --ckpt lj_ckpts/best.ckpt --Lx 4 --Ly 4 --num_particles 16 --nsamples 1024 --sample_batch_size 1024 --save lj_ckpts/samples_small.npz
