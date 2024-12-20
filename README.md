<div align="center">
<h1>Multi-Granularity Video Object Segmentation</h1>

[**Sangbeom Lim**](https://sangbeomlim.github.io/)<sup>1\*</sup> · [**Seongchan Kim**](https://github.com/deep-overflow)<sup>1\*</sup> · [**Seungjun An**](https://github.com/ansj02)<sup>3\*</sup> · [**Seokju Cho**](https://seokju-cho.github.io/)<sup>2</sup> · [**Paul Hongsuck Seo**](https://phseo.github.io/)<sup>1&dagger;</sup> . [**Seungryong Kim**](https://cvlab.korea.ac.kr)<sup>2&dagger;</sup>

<sup>1</sup>Korea University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>KAIST&emsp;&emsp;&emsp;&emsp;<sup>3</sup>Samgsung Electronics

*: Equal Contribution <br>
&dagger;: Co-Corresponding Author

**AAAI 2025**

<a href="https://arxiv.org/abs/2412.01471"><img src="https://img.shields.io/badge/arXiv-2412.01471-%23B31B1B"></a>
<a href="https://cvlab-kaist.github.io/MUG-VOS/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

<p float='center'><img src="assets/teaser.png" width="80%" /></p>
<span style="color: green; font-size: 1.3em; font-weight: bold;">MUG-VOS contains multiple granularities masks from coarse to fine segments.</span>
</div>


## 📰 News
* **2024-12-01:** [MUG-VOS](https://github.com/cvlab-kaist/MUG-VOS) is released.
* **2024-12-20:** Training Code, Data collection pipeline, and MUG-VOS Testset are released.

**Please stay tuned for a MUG-VOS v2!**

## Evaluation Dataset Preparation
First, download the evaluation datasets:
- [Huggingface Dataset](https://huggingface.co/datasets/SammyLim/MUG-VOS/tree/main/mug_vos_test_v1)

or

```bash
git clone https://huggingface.co/datasets/SammyLim/MUG-VOS
```

## Data Collection Pipeline
For detailed instructions on Data Collection Pipeline, please refer to the README file for your chosen implementation:
- **[PyTorch Implementation](./data/README.md)**


## Training and Evaluation
For detailed instructions on training and evaluation, please refer to the README file for your chosen implementation:
- **[PyTorch Implementation](./mmpm/README.md)**


## 📚 Citing this Work
Please use the following bibtex to cite our work:
```
@article{lim2024multi,
  title={Multi-Granularity Video Object Segmentation},
  author={Lim, Sangbeom and Kim, Seongchan and An, Seungjun and Cho, Seokju and Seo, Paul Hongsuck and Kim, Seungryong},
  journal={arXiv preprint arXiv:2412.01471},
  year={2024}
}
```

## 🙏 Acknowledgement
This project is largely based on the [XMem repository](https://github.com/hkchengrex/XMem). Thanks to the authors for their invaluable work and contributions.