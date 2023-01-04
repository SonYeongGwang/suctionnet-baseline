CUDA_VISIBLE_DEVICES=0 python3 inference.py --model deeplabv3plus_resnet101 \
--checkpoint_path /home/download/a/log_dir \
--split test_seen \
--camera realsense \
--dataset_root /home/download/a/test_seen \
--save_dir /home/suctionnet-baseline/save_directory \
--save_visu

