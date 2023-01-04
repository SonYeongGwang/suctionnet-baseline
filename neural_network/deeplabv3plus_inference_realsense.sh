CUDA_VISIBLE_DEVICES=0 python3 inference_realsense.py --model deeplabv3plus_resnet101 \
--checkpoint_path /home/hand-eye/suctionnet-baseline/utils/weight/realsense-deeplabplus-RGBD \
--split test_seen \
--camera realsense \
--dataset_root /home/hand-eye/suctionnet-baseline/utils/camera_information \
--save_dir /home/hand-eye/suctionnet-baseline/realsense_directory \
--save_visu

