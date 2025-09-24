<div align="center">
<h2> <img src="./imgs/logo.jpg" alt="Image Alt Text" width="50" height="50" align="absmiddle"> Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation
</h2> 
</div>
<div align="center">

[Siwei Wen](https://scholar.google.com/citations?user=kJRiUYwAAAAJ&hl=zh-CN)<sup>1,3*</sup>, 
[Junyan Ye](https://yejy53.github.io/)<sup>2,1*</sup>,
[Peilin Feng](https://peilin-ff.github.io/)<sup>1,3</sup>, 
[Hengrui Kang](https://scholar.google.com/citations?user=kVbzWCAAAAAJ&hl=zh-CN)<sup>4,1</sup>, <br>
[Zichen Wen](https://scholar.google.com/citations?user=N-aPFvEAAAAJ&hl=zh-CN)<sup>4,1</sup>, 
[Yize Chen](https://openreview.net/profile?id=~Yize_Chen2)<sup>5</sup>, 
[Jiang Wu](https://scholar.google.com/citations?user=LHiiL7AAAAAJ&hl=zh-CN)<sup>1</sup>, 
[Wenjun Wu](https://openreview.net/profile?id=~wenjun_wu3)<sup>3</sup>, 
[Conghui He](https://conghui.github.io/)<sup>1</sup>, 
[Weijia Li](https://liweijia.github.io/)<sup>2,1†</sup>

<sup>1</sup>Shanghai Artificial Intelligence Laboratory, <sup>2</sup>Sun Yat-sen University<br>
<sup>3</sup>Beihang University, <sup>4</sup>Shanghai Jiao Tong University, <sup>5</sup>The Chinese University of Hong Kong, Shenzhen

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2503.14905-AD1C18.svg?logo=arXiv)](https://arxiv.org/pdf/2503.14905) 
[![](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fopendatalab%2FFakeVLM&count_bg=%23C25AE6&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitor&edge_flat=false)](https://hits.seeyoufarm.com)
[![GitHub issues](https://img.shields.io/github/issues/opendatalab/FakeVLM?color=critical&label=Issues)](https://github.com/opendatalab/FakeVLM/issues)
[![GitHub Stars](https://img.shields.io/github/stars/opendatalab/FakeVLM?style=social)](https://github.com/opendatalab/FakeVLM/stargazers)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/lingcco/FakeClue)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/lingcco/fakeVLM)
</div>

<!-- <div align="center">
  <p align="center">
    <a href=''>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
  </p>
</div> -->

## 📰 News
- **[2025.9.24]**: 🎉 FakeVLM was accepted to Nips 2025! 
- **[2025.4.15]**: 🤗 We are excited to release the FakeClue dataset. Check out [here](https://huggingface.co/datasets/lingcco/FakeClue).
- **[2025.3.20]**: 🔥 We have released **Spot the Fake: Large Multimodal Model-Based Synthetic Image Detection with Artifact Explanation**. Check out the [paper](https://arxiv.org/abs/2503.14905). We present FakeClue dataset and FakeVLM model.

## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/599/599205.png"> FakeVLM Overview

With the rapid advancement of Artificial Intelligence Generated Content (AIGC) technologies, synthetic images have become increasingly prevalent in everyday life, posing new challenges for authenticity assessment and detection. Despite the effectiveness of existing methods in evaluating image authenticity and locating forgeries, these approaches often lack human interpretability and do not fully address the growing complexity of synthetic data. To tackle these challenges, we introduce FakeVLM, a specialized large multimodal model designed for both general synthetic image and DeepFake detection tasks. FakeVLM not only excels in distinguishing real from fake images but also provides clear, natural language explanations for image artifacts, enhancing interpretability. Additionally, we present FakeClue, a comprehensive dataset containing over 100,000 images across seven categories, annotated with fine-grained artifact clues in natural language. FakeVLM demonstrates performance comparable to expert models while eliminating the need for additional classifiers, making it a robust solution for synthetic data detection. Extensive evaluations across multiple datasets confirm the superiority of FakeVLM in both authenticity classification and artifact explanation tasks, setting a new benchmark for synthetic image detection. 

<div align="center">
<img src="imgs/framework.jpg" alt="framework" width="90%" height="auto">
</div>

## <img id="painting_icon" width="3%" src="https://cdn-icons-png.flaticon.com/256/2435/2435606.png"> Contributions

- We propose FakeVLM,  a multimodal large model designed for both general synthetic and deepfake image detection tasks. It excels at distinguishing real from fake images while also providing excellent interpretability for artifact details in synthetic images.
- We introduce the FakeClue dataset, which includes a rich variety of image categories and fine-grained artifact annotations in natural language.
- Our method has been extensively evaluated on multiple datasets, achieving outstanding performance in both synthetic detection and abnormal artifact explanation tasks.

## 🛠️ Installation
Please clone our repository and change to that folder
```bash
git clone git@github.com:opendatalab/FakeVLM.git
cd FakeVLM
```

Our model is based on the [lmms-finetune](baidu.com) environment. Please follow the steps below to configure the environment.
```bash
conda create -n fakevlm python=3.10 -y
conda activate fakevlm

python -m pip install -r requirements.txt

python -m pip install --no-cache-dir --no-build-isolation flash-attn
```

## 📦 Dataset
The directory containing the images should have the following structure:

```
playground       
└──data
    └──train
        |--doc
            |--fake
            |--real
        .
        .
        |--satellite
    └──test
        .
        .
        .    
```


## 📌 Usage
### 1. Data Preparation
The training data can be downloaded from [here](https://huggingface.co/datasets/lingcco/FakeClue).

Please download the dataset and unzip the images.
### 2. Train

Replace data paths with yours in `scripts/train.sh` and  model path with yours in `supported_models.py`.

```
bash train.sh
```

### 3. Evaluation

We prepared two scripts for you to evaluate the FakeVLM model.

#### 1. Usual evaluation 

```
bash scripts/eval.sh
```

#### 2. Evaluation with vllm

Considering the size of the model and the magnitude of the data, we recommend using vllm for evaluation. Please make sure that you have installed vllm.


```
# change scripts/eval.py to scripts/eval_vllm.py in scripts/eval.sh
bash scripts/eval.sh
```
## 📊 Results
Performance of 7 leading LMMs and FakeVLM on DD-VQA, Fake Clues and Loki.

- **FakeClue**  
  Ours dataset.
- **LOKI**  
  A new benchmark for evaluating multimodal models in synthetic detection tasks. It includes **human-annotated fine-grained image artifacts**, enabling deeper analysis of artifact explanations. We used its image modality, covering categories like Animals, Humans, Scenery, and Documents.

<img src="imgs/fakeclue_loki_result.jpg" alt="framework" width="auto" height="auto">

- **DD-VQA**  
  A dataset for explaining facial artifacts, using **manual annotations** in a VQA format. Artifacts include blurred hairlines, mismatched eyebrows, rigid pupils, and unnatural shadows. It builds on FF++ data and emphasizes common-sense reasoning.

<div align="center">
<img src="imgs/ddvqa.jpg" alt="framework" width="500" height="auto">
</div>

To provide a comprehensive comparison of the model performance across the three datasets—FakeClue, LOKI, and DD-VQA—we present the following radar chart. This chart visually highlights the strengths and weaknesses of the 7 leading LMMs and FakeVLM, offering a clear depiction of their results in synthetic detection and artifact explanation tasks.

<div align="center">
<img src="imgs/result.jpg" alt="result" width="400" height="auto">
</div>

## 😄 Acknowledgement

This repository is built upon the work of [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main), and our codebase is built upon [lmms-finetune](https://github.com/zjysteven/lmms-finetune). We appreciate their contributions and insights that have provided a strong foundation for our research.

## 📨 Contact

If you have any questions or suggestions, please feel free to contact us 
at [466439420gh@gmail.com](466439420gh@gmail.com).

## 📝 Citation
If you find our work interesting and helpful, please consider giving our repo a star. Additionally, if you would like to cite our work, please use the following format:
```shell
@article{wen2025spot,
  title={Spot the fake: Large multimodal model-based synthetic image detection with artifact explanation},
  author={Wen, Siwei and Ye, Junyan and Feng, Peilin and Kang, Hengrui and Wen, Zichen and Chen, Yize and Wu, Jiang and Wu, Wenjun and He, Conghui and Li, Weijia},
  journal={arXiv preprint arXiv:2503.14905},
  year={2025}
}
```
