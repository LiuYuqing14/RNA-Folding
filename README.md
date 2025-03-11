# Ribonanza-RNA-Folding
> Using Transformer model predicts the structures of any RNA molecule

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## General Information
- Ribonucleic acid (RNA) is essential for most biological functions, and researchers must better understand each RNA molecule's structure. Recent efforts to predict RNA structure have run into a number of challenges: (1) a paucity of training data, (2) lack of intellectual and computational power, and (3) difficulties in rigorously splitting training and test data. 
- This project is based on Transformer deep-learning model and self-attention mechanism. By taking smaller batch size, optimizing the filter of training data, and improving model implementation, the loss has reduced from 0.18 to 0.14 under similar running time.

## Technologies Used
- Tensorflow - version 2.16.1
- Python - version 3.13.0
- TPU v3-8

## Features
- experiment_type: (string) Either DMS_MaP or 2A3_MaP to describe the type of chemical mapping experiment that was used to generate each profile.
- signal_to_noise: (float) Signal/noise value for the profile
- reactivity: (float) Reactivity profile for the RNA.

## Project Status
Project is: _in progress_ 


## Room for Improvement
Room for improvement:
- The tensorflow version is not the newest version because I am on Windows, use Windows Subsystem for Linux would give a full linux environment and lower loss.
- Because of the limitation of TPU memory, the batch size used here is 128. Experimenting with smaller size would give a preciser result, but a longer runtime as a trade-off

To do:
- Feature to be added 1
- Feature to be added 2


## Acknowledgements
Give credit here.
- This project is inspired by [Mr. iafoss's work](https://www.kaggle.com/code/iafoss/rna-starter-0-186-lb/notebook) and [Mr. irohith's work](https://www.kaggle.com/code/irohith/aslfr-ctc-based-on-prev-comp-1st-place)
- This project was based on dataset from Stanford University: [this dataset](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/data).


## Contact
Created by [@LiuYuqing14](https://github.com/LiuYuqing14/Ribonanza-RNA-Folding) - feel free to contact me!
