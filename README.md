## Introduction
FreeAD is an end-to-end (E2E) robot navigation system designed to operate in unstructured environments such as auxiliary roads, campus paths, and indoor settings. Unlike traditional E2E autonomous driving models that focus on structured roads, FreeAD aims to improve navigation capabilities in these challenging scenarios.

This project introduces the FreeWorld Dataset, a comprehensive dataset combining real-world robot data and synthetic data generated using the Isaac Sim simulator. This dataset is tailored for training and evaluating E2E autonomous driving models in unstructured environments.

To validate the dataset's effectiveness, we fine-tuned the efficient E2E driving model VAD using FreeWorld. Our results demonstrate that this fine-tuning significantly enhances navigation performance in unstructured environments.

This repository provides the first dataset specifically for E2E robot navigation in unstructured scenarios and offers a benchmark for vision-based E2E navigation technology, supporting the development of logistics and service robots.

## Dataset
We modified some APIs from the nuScenes dataset to enhance flexibility and support a wider variety of data and map scenarios. The modified code has been localized and named FreeWorld.
The [FreeWorld Dataset](https://huggingface.co/datasets/doraemon6666/FreeWorld) is available for access.

## Model
The **FT-VAD** model (`FT_VAD_s1e6_s2e3.pth`) was trained for **6 epochs** in stage 1 and **3 epochs** in stage 2, based on the pre-trained [VAD-Base model](https://drive.google.com/file/d/1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS/view?usp=sharing).  
The **FT-VAD** model is available on [Hugging Face](https://huggingface.co/doraemon6666/FreeAD).


## Results

### Map Modeling Results on FreeWorld
This table presents a comparison between VAD-Tiny and VAD-Base using the Boundary + Divider map modeling strategy on the Full Warehouse map. This map structure closely aligns with the nuScenes map definition, providing a comprehensive evaluation in an open-loop scenario.
| **Method**  | **L2 (m) 1s ↓** | **L2 (m) 2s ↓** | **L2 (m) 3s ↓** | **L2 (m) Avg. ↓** | **AP Divider ↑** | **AP Boundary ↑** | **FPS** | **Collision (%) ↓** |
|:------------|:----------------:|:----------------:|:----------------:|:--------------------:|:--------------------:|:--------------------:|:-------:|:---------------------:|
| VAD-Tiny     | **1.772**         | **3.291**         | **5.008**         | **3.357**             | **0.004**             | 0.000                 | **7.6** | **0.00**              |
| VAD-Base     | 3.296             | 5.779             | 8.429             | 5.835                 | 0.001                 | 0.000                 | 4.6     | 0.00                   |

**Note:** AP Divider and AP Boundary are computed with a threshold of 1.5.  



### Open-loop Results on FreeWorld
Open-loop Results on FreeWorld: Comparison of VAD-Tiny, VAD-Base, and FT-VAD under the "Divider Only" map modeling strategy. ADE and FDE represent pedestrian motion prediction errors.
**ADE** and **FDE** represent pedestrian motion prediction errors.  

- Open-loop planning results on [FreeWorld](https://huggingface.co/datasets/doraemon6666/FreeWorld).

| **Method**  | **L2 (m) 1s ↓** | **L2 (m) 2s ↓** | **L2 (m) 3s ↓** | **Avg. ↓** | **AP divider ↑** | **FPS ↑** | **Collision (Avg. %) ↓** | **ADE ↓** | **FDE ↓** |
|:------------|:----------------:|:----------------:|:----------------:|:------------:|:------------------:|:-----------:|:---------------------------:|:------------:|:------------:|
| VAD-Tiny     | 0.891            | 1.600            | 2.449            | 1.647         | 0.000               | **8.7**     | 0.00                         | 2.848         | 3.294         |
| VAD-Base     | 0.499            | 0.759            | 1.040            | 0.766         | 0.001               | 5.0          | 0.00                         | 2.089         | 2.917         |
| **FT-VAD**   | **0.421**        | **0.596**        | **0.760**        | **0.592**     | **0.567**            | 5.0          | 0.00                         | **1.432**     | **2.319**     |

> **Note:** AP divider is computed with a threshold of 1.5.

### Open-loop Results on nuScenes
- Open-loop Planning Results on [nuScenes](https://github.com/nutonomy/nuscenes-devkit)

| **Method**  | **L2 (m) 1s ↓** | **L2 (m) 2s ↓** | **L2 (m) 3s ↓** | **Col. (%) 1s ↓** | **Col. (%) 2s ↓** | **Col. (%) 3s ↓** |
|:------------|:----------------:|:----------------:|:----------------:|:--------------------:|:--------------------:|:--------------------:|
| VAD-Tiny     | 0.46             | 0.76             | 1.12             | 0.21                 | 0.35                 | 0.58                 |
| VAD-Base     | **0.41**         | **0.70**         | **1.05**         | 0.07                 | 0.17                 | 0.41                 |
| **FT-VAD**   | 2.02             | 3.24             | 4.38             | **0.00**             | **0.00019**          | **0.00016**          |

## Getting Started
- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)
- [Train and Eval](docs/train_eval.md)
- [Visualization](docs/visualization.md)

## Contact
If you have any questions or suggestions about this repo, please feel free to contact us (yuhangpeng65@gmail.com).

## License
All code in this repository is under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
