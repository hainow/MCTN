This repo implements the Multimodal Cyclic Translation Network (MCTN) model 
for the following paper:
 
###### Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities
Hai Pham*, Paul Pu Liang*, Thomas Manzini, Louis-Philippe Morency, Barnabás Poczós. AAAI 2019. 

### Installation 
You need to have numpy and the following standard packages which can also be 
easily installed using `pip`. 

`tensorflow>=1.4.0` 

`keras>=2.1.2`

`recurrentshop`


We also need to mention that the seq2seq code is extended from the following 
github: https://github.com/farizrahman4u/seq2seq. 
We are grateful for this great repo which really 
helped us in speeding up our implementation and experiments. 

### Usage

First add the current directory to the $PYTHONPATH by `source set_up`. 
Second, you need to process your data according to the 
instruction in the function `utils/data_loader
.py/load_and_preprocess_data`. To use the code directly, you need a 3D Numpy 
array for each modality. After that: 

For running the Bimodal MCTN:
```bash
$ python enc2end_bimodal.py \ 
    --train_epoch 200 \ 
    --batch_size 32 \ 
    --feature t f \ 
    --cfg configs/mctn.yaml 
```

For running the Trimodal (Hierarchical) MCTN:
```bash
$ python enc2end_trimodal.py \ 
    --train_epoch 200 \ 
    --batch_size 32 \ 
    --feature t f c \ 
    --cfg configs/hierarchical_mctn.yaml 
```

Note that you can also run those scripts directly with our default arguments. 
For changing 
those arguments, please refer to `args.py` in the `utils` package 
for general arguments. For architecture specific settings, please extend from
 the sample configuration files in the `configs` directory. Furthermore, 
 you can easily follow our standard models in the packages `models` to 
 design new architecture for your specific use case. 

### License 

Standard GPL License. See the LICENSE file for more detail. 

Copyright 2019 Hai Pham. 

### Citation
 
If you use any part of this code in your paper, please cite our [paper](https://arxiv.org/abs/1812.07809)
```angular2html
@article{pham2018found,
  title={Found in Translation: Learning Robust Joint Representations by Cyclic Translations Between Modalities},
  author={Pham, Hai and Liang, Paul Pu and Manzini, Thomas and Morency, Louis-Philippe and Poczos, Barnabas},
  journal={arXiv preprint arXiv:1812.07809},
  year={2018}
}
```



