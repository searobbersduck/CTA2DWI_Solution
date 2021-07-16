CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.launch \
--master_addr='10.100.37.100' \
--master_port='29507' \
--nproc_per_node=2 \
--nnodes=1 \
--use_env \
train.py \
--dataroot /data/medical/brain/gan/cta2dwi_multi_classified/5.train_batch \
--model pix2pix_3d \
--input_nc 1 \
--output_nc 1 \
--ngf 32 \
--netG resnet_6blocks \
--ndf 8 \
--no_dropout \
--netD pixel \
--norm batch \
--display_server='10.100.37.100' \
--display_port=8901 \
--display_id=0 \
--lambda_L1=1 \
--n_epochs=500 \
--display_freq=10 \
--print_freq=10 \
--save_epoch_freq=10 \
--lr_policy cosine \
--lr 1e-4 \
--checkpoints_dir /data/medical/brain/gan/cta2dwi_multi_classified/checkpoints \
--name cta2dwi_sr \
--crop_size 384 384 160 \
--src_pattern fixed_cta.nii.gz \
--dst_pattern registried_dwi_parenchyma.nii.gz \
--mask_pattern final_mask.nii.gz \
--mask_label 1 \
--lambda_L1_Mask 1.0 \
--no_discriminator \
--ssl_sr \
--ssl_arch resnet10 \
--ssl_pretrained_file /data/medical/brain/gan/cta2dwi_multi_classified/ssl/ \
--continue_train