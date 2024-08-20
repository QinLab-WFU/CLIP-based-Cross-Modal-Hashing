# CLIP-based Cross-Modal Hashing


## Method
[DCHMT](https://dl.acm.org/doi/10.1145/3503161.3548187) [ACM MM 2022] [Source Code](https://github.com/kalenforn/DCHMT)  
[DSPH](https://ieeexplore.ieee.org/document/10149001) [TCSVT 2023] [Source Code](https://github.com/QinLab-WFU/DSPH)  
[MITH](https://dl.acm.org/doi/10.1145/3581783.3612411) [ACM MM 2023] [Source Code](https://github.com/DarrenZZhang/MM23-MITH)  
[DNpH](https://ieeexplore.ieee.org/document/10379137) [TMM 2024] [Source Code](https://github.com/QinLab-WFU/DNpH)  
[DNPH](https://dl.acm.org/doi/abs/10.1145/3643639) [TOMM 2024] [Source Code](https://github.com/QinLab-WFU/OUR-DNPH)  
[TwDH](https://ieeexplore.ieee.org/document/10487033) [TMM 2024] [Source Code](https://github.com/kalenforn/clip-based-cross-modal-hash/tree/main/runners/TwDH)  
[DHaPH](https://ieeexplore.ieee.org/abstract/document/10530441) [TKDE 2024] [Source Code](https://github.com/QinLab-WFU/DHaPH)  


## Processing dataset
Before training, you need to download the oringal data from [coco](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)(include 2017 train,val and annotations), [nuswide](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)(include all), mirflickr25k [Baidu, 提取码:u9e1](https://pan.baidu.com/s/1upgnBNNVfBzMiIET9zPfZQ) or [Google drive](https://drive.google.com/file/d/18oGgziSwhRzKlAjbqNZfj-HuYzbxWYTh/view?usp=sharing) (include mirflickr25k and mirflickr25k_annotations_v080), 
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

## Start

for example


DSPH:
> python main.py --method DSPH --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --numclass 80

 
TwDH:
> python main.py --method TwDH --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 512 --save-dir ./result/coco/512 --clip-path ./ViT-B-32.pt --batch-size 128


## Acknowledegements
[DCHMT](https://github.com/kalenforn/DCHMT)  
[DSPH](https://github.com/QinLab-WFU/DSPH)  
[MITH](https://github.com/DarrenZZhang/MM23-MITH)  
[DNpH](https://github.com/QinLab-WFU/DNpH)  
[DNPH](https://github.com/QinLab-WFU/OUR-DNPH)  
[TwDH](https://github.com/kalenforn/clip-based-cross-modal-hash/tree/main/runners/TwDH)  
[DHaPH](https://github.com/QinLab-WFU/DHaPH)  
