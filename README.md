### TRACER
### Tumor aware recurrent inter‐patient deformable image registration of computed tomography scans with lung cancer
Code for TRACER Deformable image registration method
The code for the paper, **Tumor aware recurrent inter-patient deformable image registration of computed tomography scans with lung cancer**, by Jue Jiang, Chloe Min-Seo Choi, Maria Thor, Joseph O. Deasy, and Harini Veeraraghavan published in the Medical Physics, 2024 is here. 
The paper is available https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.17536 and https://arxiv.org/abs/2409.11910


![image](https://github.com/user-attachments/assets/38c96ada-6ac3-4f8c-80f3-9f8084b6a455)



## Prerequisites
- Linux
- Python 3.9
- NVIDIA GPU with CUDA CuDNN (CUDA 11.8 or higher)

## To get started
- Clone this repository
- Install pytorch (our GPU is at CUDA 11.8, so we use the following command `pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118`)
- Install the requirements using `pip install -r requirements.txt`
- The data should be present in the `datadir` folder, with the json organizing each instance under `train` and `validation` keys respectively
- For training:
    - Run the training script using the following command:
        ```bash
        python train.py \
        --datadir {path to datadir} \
        --json {json_name} \
        --image_loss {loss function to use} \
        --cache {Load images in memory}
        ```
 ## Note

## Citation
If you use this code for your research, please cite our paper <a href="https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.17536">TRACER</a>:

```
@article{jiang2024tumor,
  title={Tumor aware recurrent inter-patient deformable image registration of computed tomography scans with lung cancer},
  author={Jiang, Jue and Choi, Chloe Min Seo and Thor, Maria and Deasy, Joseph O and Veeraraghavan, Harini},
  journal={Medical Physics},
  year={2024},
  publisher={Wiley Online Library}
}
```
