# Baseline Methods for SuctionNet-1Billion

You can use your own camera on baseline methods in RA-L paper "SuctionNet-1Billion:  A  Large-Scale  Benchmark  for  Suction  Grasping" 


## Camera

Only realsense camera and kinect camera are required

## Environment

The code has been tested with `CUDA 10.1` and `pytorch 1.4.0` on ubuntu `16.04`
Others are on requirement.txt

*torch/torch vision install
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

## Usage

### Neural Networks

Change the directory to `neural_network`:

```
cd neural_network
```
For inference, use the following command: 

```
python inference.py \
--model model_name \
--checkpoint_path /path/to/the/saved/model/weights \
--camera realsense \ # realsense or kinect
--save_dir /path/to/save/the/inference/results \
--save_visu # whether to save the visualizations
```

or modify [scripts/deeplabv3plus_inference_kinect.sh](https://github.com/heechan99/suctionnet-baseline/blob/master/neural_network/deeplabv3plus_inference_kinect.sh), [scripts/deeplabv3plus_inference_realsense.sh](https://github.com/heechan99/suctionnet-baseline/blob/master/neural_network/inference_realsense.py), to inference with your kinect and realsense camera.
### Normal STD

Change the directory to `normal_std` by:

```
cd normal_std
```

This method does not need to train, you can directly inference with the following command:

```
python inference.py 
--split test_seen \ # can be test, test_seen, test_similar, test_novel
--camera realsense \ # realsense or kinect
--save_root /path/to/save/the/inference/results \
--dataset_root /path/to/SuctionNet/dataset \
--save_visu
```

or modify [inference.sh](https://github.com/graspnet/suctionnet-baseline/blob/master/normal_std/inference.sh) and run `sh inference.sh`

## Pre-trained Models

### RGB-D Models

We provide models including [our model for realsense](https://drive.google.com/file/d/18TbctdhpNXEKLYDWFzI9cT1Wnhe-tn9h/view?usp=sharing), [our model for kinect](https://drive.google.com/file/d/1gOz_KmIugBGUtpcyHAgYO01T0h5ZqOl9/view?usp=sharing), [Fully Conv Net for realsense](https://drive.google.com/file/d/1hgYYIvw5Xy-r5C8IitKizswtuMV_EqPP/view?usp=sharing) ,[Fully Conv Net for kinect](https://drive.google.com/file/d/1A6K5EmItBuDaxrWyz5g8zSHY5Kw1_NnX/view?usp=sharing).

### Depth Models

Our models only taking in depth images are also provided [for realsense](https://drive.google.com/file/d/1q2W2AV663PNT4_TYo5zZtYxjenZJ7GAb/view?usp=sharing) and [for kinect](https://drive.google.com/file/d/1mAzFC9dlEDBuoHQp7JGTcTkKGSwFnVth/view?usp=sharing).

## Citation

if you find our work useful, please cite

```
@ARTICLE{suctionnet,
  author={Cao, Hanwen and Fang, Hao-Shu and Liu, Wenhai and Lu, Cewu},
  journal={IEEE Robotics and Automation Letters}, 
  title={SuctionNet-1Billion: A Large-Scale Benchmark for Suction Grasping}, 
  year={2021},
  volume={6},
  number={4},
  pages={8718-8725},
  doi={10.1109/LRA.2021.3115406}}
```

