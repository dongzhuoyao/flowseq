#  Flow Matching for Conditional Text Generation in a Few Sampling Steps

**EACL 2024**

This repository represents the official implementation of the EACL2024 paper titled "Flow Matching for Conditional Text Generation in a Few Sampling Steps".

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/taohu/flowsea)
[![Website](doc/badges/badge-website.svg)](https://taohu.me/project_flowseq)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://aclanthology.org/2024.eacl-short.33.pdf)
[![GitHub](https://img.shields.io/github/stars/dongzhuoyao/flowseq?style=social)](https://github.com/dongzhuoyao/flowseq)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)


 <span class="author-block">
<a href="https://taohu.me/" target="_blank">Vincent Tao Hu,</a></span>
<span class="author-block">
<a href="https://moore3930.github.io/" target="_blank">Di Wu,</a></span>
<span class="author-block">
  <a href="https://yukimasano.github.io/" target="_blank">Yuki M. Asano,</a>
</span>
<span class="author-block">
  <a href="https://staff.fnwi.uva.nl/p.s.m.mettes/" target="_blank">Pascal Mettes,</a>
</span>
<span class="author-block">
  <a href="https://basurafernando.github.io/" target="_blank">Basura Fernando,</a>
</span>
<span class="author-block">
  <a href="https://scholar.google.de/citations?user=zWbvIUcAAAAJ&hl=en" target="_blank"> Bjorn Ommer, </a>
</span>

<span class="author-block">
  <a href="https://www.ceessnoek.info/" target="_blank">Cees G.M. Snoek</a>
</span>



# Dataset

```
https://drive.google.com/drive/folders/1sU8CcOJE_aaaKLijBNzr-4y1i1YoZet2?usp=drive_link
```

## Run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4 flow_train.py 
CUDA_VISIBLE_DEVICES=0,2 torchrun --nnodes=1 --nproc-per-node=2 flow_train.py  data=qg
CUDA_VISIBLE_DEVICES=6 torchrun --nnodes=1 --nproc-per-node=1 flow_train.py  data=qg
```


# Evaluation 

Download the pretrained checkpoint from [https://huggingface.co/taohu/flowseq/tree/main](https://huggingface.co/taohu/flowseq/tree/main), more checkpoints are coming soon.



```python
python flow_sample_eval_s2s.py    data=qqp_acc data.eval.is_debug=0 data.eval.model_path='qqp_ema_0.9999_070000.pt' data.eval.candidate_num=1 data.eval.ode_stepnum=1
```


# Environment Preparation

```bash
conda create -n flowseq  python=3.10
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install  torchdiffeq  matplotlib h5py  accelerate loguru blobfile ml_collections
pip install hydra-core wandb einops scikit-learn --upgrade
pip install einops 
pip install transformers
pip install nltk bert_score datasets torchmetrics
```

Optional
```bash
pip install diffusers
```



# Common Issue

Typical Issue:

```bash 
https://github.com/Shark-NLP/DiffuSeq/issues/5
https://github.com/Shark-NLP/DiffuSeq/issues/22
```




## Citation
Please add the citation if our paper or code helps you.

```
@inproceedings{HuEACL2024,
        title = {Flow Matching for Conditional Text Generation in a Few Sampling Steps},
        author = {Vincent Tao Hu and Di Wu and Yuki M Asano and Pascal Mettes and Basura Fernando and Björn Ommer and Cees G M Snoek},
        year = {2024},
        date = {2024-03-27},
        booktitle = {EACL},
        tppubtype = {inproceedings}
        }
```
