# Edge-Direct Visual Odometry
If you find this useful, please cite the related [paper](https://arxiv.org/pdf/1906.04838.pdf):
```
@article{DBLP:journals/corr/abs-1906-04838,
  author    = {Kevin Christensen and
               Martial Hebert},
  title     = {Edge-Direct Visual Odometry},
  journal   = {CoRR},
  volume    = {abs/1906.04838},
  year      = {2019},
  url       = {http://arxiv.org/abs/1906.04838},
  archivePrefix = {arXiv},
  eprint    = {1906.04838},
}
```
## Setup
This repository assumes the following directory structure, and is setup for the [TUM-RGBD Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset):
```
EdgeDirectVO
|-- tum_rgbd_dataset/
    |-- depth/
    |-- rgb/
    |-- assoc.txt
|-- tum_rgbd_dataset_ground_truth.txt
```

Be sure to run [assoc.py](https://vision.in.tum.de/data/datasets/rgbd-dataset) to associate timestamps with corresponding frames.

## Build and Run
```
cd build && cmake .. && make -j && ./EdgeDirectVO
```

## Evaluation
After evaluating on a dataset, the corresponding evaluation commands will be printed to terminal.  Simpy copy and run them in terminal in project root directory.  Examples are shown below:

### For [relative pose error](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools):
```
python evaluate_rpe.py groundtruth_fr1xyz.txt rgbd_dataset_freiburg1_xyz_results.txt --fixed_delta --delta 1 --verbose
```
### For [absolute trajectory error](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools):
```
python evaluate_ate.py groundtruth_fr1xyz.txt rgbd_dataset_freiburg1_xyz_results.txt --verbose
```