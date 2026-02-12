# CLIP-based Cross-Modal Hashing


## Method
[DCHMT](https://dl.acm.org/doi/10.1145/3503161.3548187) [ACM MM 2022] [Source Code](https://github.com/kalenforn/DCHMT)  
[DSPH](https://ieeexplore.ieee.org/document/10149001) [TCSVT 2023] [Source Code](https://github.com/QinLab-WFU/DSPH)  
[MITH](https://dl.acm.org/doi/10.1145/3581783.3612411) [ACM MM 2023] [Source Code](https://github.com/DarrenZZhang/MM23-MITH)  
[DNpH](https://ieeexplore.ieee.org/document/10379137) [TMM 2024] [Source Code](https://github.com/QinLab-WFU/DNpH)  
[DNPH](https://dl.acm.org/doi/abs/10.1145/3643639) [TOMM 2024] [Source Code](https://github.com/QinLab-WFU/OUR-DNPH)  
[TwDH](https://ieeexplore.ieee.org/document/10487033) [TMM 2024] [Source Code](https://github.com/kalenforn/clip-based-cross-modal-hash/tree/main/runners/TwDH)  
[DHaPH](https://ieeexplore.ieee.org/abstract/document/10530441) [TKDE 2024] [Source Code](https://github.com/QinLab-WFU/DHaPH)  
[DMsH-LN](https://www.sciencedirect.com/science/article/pii/S0925231224016011) [Neucomputing 2024] [Source Code](https://github.com/QinLab-WFU/DMsH-LN)  
[DScPH](https://ieeexplore.ieee.org/document/10855579) [TMM 2025] [Source Code](https://github.com/QinLab-WFU/DScPH)  
[DNcH](https://www.sciencedirect.com/science/article/abs/pii/S095741742500987X) [ESWA 2025] [Source Code](https://github.com/QinLab-WFU/DNcH)  
[DDBH](https://ieeexplore.ieee.org/document/11003934) [TCSVT 2025] [Source Code](https://github.com/QinLab-WFU/DDBH)  
[DAGtH](https://www.sciencedirect.com/science/article/abs/pii/S0957417425021852) [ESWA 2025] [Source Code](https://github.com/QinLab-WFU/OUR-DAGtH)  
[DPBE](https://dl.acm.org/doi/10.1145/3746027.3754811) [ACM MM 2025] [Source Code](https://github.com/QinLab-WFU/DPBE)  
[DDWSH](https://ieeexplore.ieee.org/document/11353914) [TMM 2026] [Source Code](https://github.com/QinLab-WFU/DDWSH)  
DPSIH [AAAI 2026] [Source Code](https://github.com/QinLab-WFU/DPSIH)  
DGHDGH [ICLR 2026] [Source Code](https://github.com/QinLab-WFU/DGHDGH)  


## Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), nuswide [Google drive](https://drive.google.com/file/d/11w3J98uL_KHWn9j22GeKWc5K_AYM5U3V/view?usp=drive_link), mirflickr25k [Baidu, 提取码:u9e1](https://pan.baidu.com/s/1upgnBNNVfBzMiIET9zPfZQ) or [Google drive](https://drive.google.com/file/d/18oGgziSwhRzKlAjbqNZfj-HuYzbxWYTh/view?usp=sharing) (include mirflickr25k and mirflickr25k_annotations_v080), 
then use the "data/make_XXX.py" to generate .mat file

After all mat file generated, the dir of `dataset` will like this:
~~~
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Notice! It is a txt file!
    ├── index.mat 
    └── label.mat
~~~

## Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

## Prepare Stochman for DPBE
> cd ./train/DPBE/stochman/
> 
> python setup.py install
You may need to manually adjust the import path after installing locally.

## Training

> python main.py --method DCHMT --dataset coco --lr 0.001 --output-dim 64 --clip-path ./ViT-B-32.pt

## Acknowledegements
[DCHMT](https://github.com/kalenforn/DCHMT)  
[DSPH](https://github.com/QinLab-WFU/DSPH)  
[MITH](https://github.com/DarrenZZhang/MM23-MITH)  
[DNpH](https://github.com/QinLab-WFU/DNpH)  
[DNPH](https://github.com/QinLab-WFU/OUR-DNPH)  
[TwDH](https://github.com/kalenforn/clip-based-cross-modal-hash/tree/main/runners/TwDH)  
[DHaPH](https://github.com/QinLab-WFU/DHaPH)  
[DMsH-LN](https://github.com/QinLab-WFU/DMsH-LN)  
[DScPH](https://github.com/QinLab-WFU/DScPH)  
[DDBH](https://github.com/QinLab-WFU/DDBH)  
[DPBE](https://github.com/QinLab-WFU/DPBE)  
[DDWSH](https://github.com/QinLab-WFU/DDWSH)  
[DPSIH](https://github.com/QinLab-WFU/DPSIH)  
[DGHDGH](https://github.com/QinLab-WFU/DGHDGH)  
