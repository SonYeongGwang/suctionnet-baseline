CUDA_VISIBLE_DEVICES=0 python3 inference_kinect.py --model deeplabv3plus_resnet101 \
--checkpoint_path /home/hand-eye/suctionnet-baseline/weight/kinect-deeplabplus-RGBD \
--camera kinect \
--save_dir /home/hand-eye/suctionnet-baseline/save_directory \
--save_visu

