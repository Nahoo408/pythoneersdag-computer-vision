# Computer Vision

This Pythoneersday we will be diving into computer vision.

## Getting started

Clone the project and install the dependencies with the following command: 

```bash
poetry install
```

Then run this command to start the jupyter notebook server:

```bash
poetry run jupyter lab 
```

After some startup time, your browser should open on http://localhost:8888/lab. This interactive environment can be used for this workshop.

## Project layout

The project contains three different notebooks (found in `/notebooks`), one for each part of this workshop. Every section will start with a 10-15 min presentation where we explain some theory. Thereafter, you can work on the assignments alone or in small groups. Most notebooks require some additional data, which can be found in `/data`. Also, we have created some utility functions. These can be found in `/utils`, we encourage you to use them to save time and stay away from some tedious implementation details.

## Tips: 
If you want to see line numbers in the notebook, you can enable them by going to the menu bar and selecting View > Show Line Numbers.

## 1. Object Detection using hand-crafted features

Open `/notebooks/1_hand_crafted_features.ipynb` and follow it.

## 2. Object Detection by automating features

Open `/notebooks/2_automating_features.ipynb` and follow it.

## 3. Object Detection using neural networks

Open `/notebooks/3_neural_networks.ipynb` and follow it.