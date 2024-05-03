# AdvNotRealFeatures

This repository contains the official code to reproduce the main results from the NeurIPS 2023 paper titled [Adversarial Examples Are Not Real Features](https://arxiv.org/abs/2310.18936) by Ang Li*, [Yifei Wang*](https://yifeiwang77.com), [Yisen Wang](yisenwang.github.io).

In this paper, we generalize the definition of feature usefulness and robustness to multiple paradigms. This repository thus contains the code of four paradigms considered in the paper, CL (Contrastive Learning), DM (Diffusion Model), MIM (Masked Image Modeling), and SL (Supervised Learning), as subfolders. 

### Requirements
- Basically, we run with PyTorch==1.13 and Python==3.8
- To build the environment:

```
    pip install -r requirements.txt 
```

### Usage
- The whole evaluation pipeline used in our paper is defined in **run.sh**
- After building the environment, running the evaluation
```
    sh run.sh
```
- To use the pre-generated datasets
    - Download the datasets via this [link](https://drive.google.com/drive/folders/11IQ9AvKV22RGffJcEyJdoiS1i1enW__0?usp=drive_link)
    - Place the datasets into ./data folder
    - Set $EXISTING_DATASET in run.sh as TRUE
    - run the above command

- To generate the robust/non-robust datasets
    - Weights are also available at [link](https://drive.google.com/drive/folders/11IQ9AvKV22RGffJcEyJdoiS1i1enW__0?usp=drive_link)
    - Feel free to experiment with models trained with different paradigms/algorithms.

- SL
    - Implementation of Supervised Learning 
    - Model is defaulted to ResNet-18
    - See ./SL/train_cl.py for more details

- MIM
    - Implementation of the Masked Image Modeling and its linear probing
    - Model is defaulted to MAE with ViT-t
    - See ./MIM/train_mim.py for more details

- CL
    - Implementation of the Contrastive Learning its linear probing
    - Model is defaulted to SimCLR with ResNet-18
    - See ./CL/train_cl.py for more details

- DM
    - Implementation of the Diffusion Model its linear proing
    - Model is defaulted to DDPM with UNet
    - See ./DM/train_cm.py for more details

- Transfer
    - Implementation of the study of paradigm-wise transferability of non-robust features
    - Place the SimCLR.pt and SimCLR_Classifier.pt downloaded from the [link](https://drive.google.com/drive/folders/11IQ9AvKV22RGffJcEyJdoiS1i1enW__0?usp=drive_link) into this folder
    - Enjoy experimenting with the notebook

- Robust 
    - Implementation of the Valina Adversarial Training Algorithm and PGD, AA attacks
    - To install the attacks, simply run
    ```
        pip install torchattacks
        pip install -e ./Robust/attack/auto-attack
    ```
    - See eval.py in the folder as an example

### Contact Us
- Having unresolved questions about our work or feeling to have a discussion with the authors?
- Feel free to contact us! Emails of the authors are listed below:
    - Ang Li: charles_li@stu.pku.edu.cn
    - Yifei Wang: mailto:yifei_wang@pku.edu.cn
    - Yisen Wang: yisen.wang@pku.edu.cn

### Cite our work
ArXsiv Version:
```
@article{li2023adversarial,
    title={Adversarial examples are not real features},
    author={Li, Ang and Wang, Yifei and Guo, Yiwen and Wang, Yisen},
    journal={arXiv preprint arXiv:2310.18936},
    year={2023}
}
```

Conference Version:
```
@inproceedings{li2023advnotrealfeatures,
    title={Adversarial Examples Are Not Real Features},
    author={Li, Ang and Wang, Yifei and Guo, Yiwen and Wang, Yisen},
    booktitle={NeurIPS},
    year={2023}
}
```


### Acknowledgement

This repo is partially based upon the following repos and we sincerely appreciate their excellent works!
- [https://github.com/IcarusWizard/MAE](https://github.com/IcarusWizard/MAE)
- [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)
- [https://github.com/tqch/ddpm-torch](https://github.com/tqch/ddpm-torch)
- [https://github.com/MadryLab/constructed-datasets](https://github.com/MadryLab/constructed-datasets)



