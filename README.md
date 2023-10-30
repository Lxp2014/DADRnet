# DADRnet
codes of “DADRnet: Cross-domain Image Dehazing via Domain Adaptation and Disentangled Representation”

environment: conda env create -f DRDA.yaml 

test code：
python test.py --Quality_restore --use_v2 --test_input test_images/URHI/ --outputs_dir output/URHI/ --gpu_ids 0
train code：
python train.py --training_dataset domain_A --name domainA_SR_old_photos --loadSize 256 --fineSize 256 --dataroot /data/lxp/haze/dataset/ --resize_or_crop crop_only --batchSize 8 --no_html --gpu_ids 0,1,2,3 --start_r 1 --outputs_dir out --checkpoints_dir checkpoints --display_freq 100 --save_latest_freq 5 --no_cgan --lambda_DC 5 --lambda_TV 0.01 --lr_D 1
