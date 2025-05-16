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
[DNcH](https://www.sciencedirect.com/science/article/abs/pii/S095741742500987X) [Expert Systems with Applications 2025] [Source Code](https://github.com/QinLab-WFU/DNcH)  

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

## Training


First, the 'method' parameter needs to be changed in main.py. Then, run the following command.

DCHMT:
> python main.py --method DCHMT --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/DCHMT/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --similarity-function euclidean --loss-type l2 --vartheta 0.5 --sim-threshold 0.1


DSPH:
> python main.py --method DSPH --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/DSPH/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --numclass 80 --alpha 0.8


MITH:
> python main.py --method MITH --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/MITH/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128


DNpH
> python main.py --method DNpH-TMM --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/DNpH-TMM/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128


DNPH
> python main.py --method DNPH-TOMM --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/DNpH-TMM/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --numclass 80

 
TwDH:
> python main.py --method TwDH --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 512 --save-dir ./result/TwDH/coco/512 --clip-path ./ViT-B-32.pt --batch-size 128 --long_center ./train/TwDH/center/coco/long --short_center ./train/TwDH/center/coco/short --trans_matrix ./train/TwDH/center/coco/trans


DHaPH
> python main.py --method DHaPH --is-train --dataset coco --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 64 --save-dir ./result/DHaPH/coco/64 --clip-path ./ViT-B-32.pt --batch-size 128 --HM 500 --margin 0.1 --topk 15 --alpha 1 --tau 0.3


DMsH-LN:
> python main.py --method DMsH-LN --is-train --dataset flickr25k --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 16 --save-dir ./result/DMsH-LN/flickr25k/16 --clip-path ./ViT-B-32.pt --batch-size 128 --numclass 24


DScPH:
> python main.py --method DScPH --is-train --dataset flickr25k --caption-file caption.mat --index-file index.mat --label-file label.mat --lr 0.001 --output-dim 16 --save-dir ./result/DScPH/flickr25k/16 --clip-path ./ViT-B-32.pt --batch-size 128 --numclass 24
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

