<div align="center">
  <h1>VGGT-SLAM 2.0</h1>

  <p>
    <strong>VGGT-SLAM</strong> 
    <a href="https://arxiv.org/abs/2505.12549">
      <img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" alt="arXiv" style="vertical-align:middle">
    </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <strong>VGGT-SLAM 2.0</strong> 
    <a href="https://arxiv.org/abs/2601.19887">
      <img src="https://img.shields.io/badge/arXiv-b33737?logo=arXiv" alt="arXiv" style="vertical-align:middle">
    </a>
  </p>

  <br />

  <img src="assets/vggt_slam_demo.gif" alt="VGGT-SLAM" width="95%"/>

  <p><strong><em>VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction</em></strong></p>

  <p>
    <a href="https://dominic101.github.io/DominicMaggio/"><strong>Dominic Maggio</strong></a> &nbsp;·&nbsp;
    <a href="https://lucacarlone.mit.edu/"><strong>Luca Carlone</strong></a>
  </p>
</div>

---

# This repo contains the code for VGGT-SLAM 2.0 (located here) and VGGT-SLAM (located on the version1.0 branch of this repo).

## 📚 Table of Contents
* [💻 Installation](#installation-of-vGGT-sLAM)
* [🚀 Quick Start](#quick-start)
* [📊 Running Evaluations](#running-evaluations)
* [📄 News and Updates](#News-and-Updates)
* [📄 Paper Citation](#citation)

---

## Installation of VGGT-SLAM

Clone VGGT-SLAM:

```
git clone https://github.com/MIT-SPARK/VGGT-SLAM
```

```
cd VGGT-SLAM
```

### Create and activate a new conda environment

```
conda create -n vggt-slam python=3.11
```

```
conda activate vggt-slam
```

### Make the setup script executable and run it
This step will automatically download all 3rd party packages including Perception Encoder, SAM 3, and our fork of VGGT. More details on the license for Perception Encoder 
can be found [here](https://github.com/facebookresearch/perception_models/blob/main/LICENSE.PE), for SAM3 can be found [here](https://github.com/facebookresearch/sam3/blob/main/LICENSE), and for VGGT can be found [here](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt). Note that we only use SAM 3 and Perception Encoder for optional open-set 3D object detection.

```
chmod +x setup.sh
./setup.sh
```

---

## Quick Start

run `python main.py --image_folder /path/to/image/folder --max_loops 1 --vis_map` replacing the image path with your folder of images. 
This will create a visualization in viser which shows the incremental construction of the map.

As an example, we provide a folder of test images in `office_loop.zip` which will generate the following map. Using the default parameters will
result in a single loop closure towards the end of the trajectory. Unzip the folder and set its path as the arguments for `--image_folder`, e.g.,

```
unzip office_loop.zip
```

and then run the below command:

```
python3 main.py --image_folder office_loop --max_loops 1 --vis_map
```

Use the `--run_os` flag to enable 3D open-set object detection. This will prompt the user for text queries and plot a 3D bounding box of the detection on the map in viser. The office loop scene does not have very many interesting objects, but some example queries that can be used are "coffee machine", "sink", "printer", "cone", and "refrigerator." For some example scenes with more interesting objects, check out the Clio apartment and cubicle scene which can be downloaded 
from [here](https://www.dropbox.com/scl/fo/5bkv8rsa2xvwmvom6bmza/AOc8VW71kuZCgQjcw_REbWA?rlkey=wx1njghufcxconm1znidc1hgw&e=1&st=c809h8h3&dl=0).

<p align="center">
  <img src="assets/office_loop_figure.png" width="300">
</p>


### Collecting Custom Data

To quickly collect a test on a custom dataset, you can record a trajectory with a cell phone and convert the MOV file to a folder of images with:

```
mkdir <desired_location>/img_folder
```

And then, run the command below:

```
ffmpeg -i /path/to/video.MOV -vf "fps=10" <desired_location>/img_folder/frame_%04d.jpg
```
Note while vertical cell phone videos can work, to avoid images being cropped it is recommended to use horizontal videos. 

### Adjusting Parameters

See main.py or run `--help` from main.py to view all parameters. 

For visualizing larger datasets, displaying all 3D points in Viser can either slow or crash 
the visualizer. One way to mitigate this is to sparsify the point cloud that is sent to Viser which can be done with `--vis_voxel_size 0.005`. Increasing the number will decrease the number of displayed points. Note that this does not affect the number of points stored or used internally in VGGT-SLAM.

---

## Running Evaluations

To automatically run evaluation on TUM and 7-Scenes datasets, first install the datasets using the provided download instructions from [MASt3R-SLAM](https://github.com/rmurai0610/MASt3R-SLAM?tab=readme-ov-file#examples). Set the download location of MASt3R-SLAM by setting *abs_dir* in the bash scripts 
*/evals/eval_tum.sh* and */evals/eval_7scenes.sh*

#### In Tum Dataset

To run on TUM, run `./evals/eval_tum.sh <w>` and then run `python evals/process_logs_tum.py --submap_size <w>` to analyze and print the results, where w is 
the submap size, for example:

```
./evals/eval_tum.sh 32
```

```
python evals/process_logs_tum.py --submap_size 32
```

To visualize the maps as they being constructed, inside the bash scripts add `--vis_map`. This will update the viser map each time the submap is updated. 

## News and Updates
* May 2025: VGGT-SLAM 1.0 is released
* August 2025: SL(4) optimization is integrated into the official GTSAM repo
* September 2025: VGGT-SLAM 1.0 Accepted to Neurips 2025
* November 2025: VGGT-SLAM 1.0 Featured in MIT News [article](https://news.mit.edu/2025/teaching-robots-to-map-large-environments-1105)
* January 2026: VGGT-SLAM 2.0 is released

## Todo

- [ ] Release real-time code. This code enables plugging in a Real Sense Camera and incrementally constructing a map 
as the camera explored a scene. This has been tested on a Jetson Thor onboard a robot.
- [ ] Add optional code to sparsify the visualized map as visualizing large point cloud maps can slow down the code.

## Acknowledgement

This work was supported in part by the NSF Graduate Research Fellowship
Program under Grant 2141064, the ARL DCIST program, and the ONR
RAPID program.

## Citation

If our code is helpful, please cite our papers as follows:

```
@article{maggio2025vggt-slam,
  title={VGGT-SLAM: Dense RGB SLAM Optimized on the SL (4) Manifold},
  author={Maggio, Dominic and Lim, Hyungtae and Carlone, Luca},
  journal={Advances in Neural Information Processing Systems},
  volume={39},
  year={2025}
}
```

```
@article{maggio2025vggt-slam2,
  title={VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction},
  author={Maggio, Dominic and Carlone, Luca},
  journal={arXiv preprint arXiv:2601.19887},
  year={2026}
}
```

