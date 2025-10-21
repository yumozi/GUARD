# Towards Adversarially Robust Dataset Distillation by Curvature Regularization (AAAI 2025)

Official implementation of paper:
>[__"Towards Adversarially Robust Dataset Distillation by Curvature Regularization"__](https://arxiv.org/abs/2403.10045)<br>
>Eric Xue, Yijiang Li, Haoyang Liu, Peiran Wang, Yifan Shen, Haohan Wang<br>
[`[Paper]`](https://arxiv.org/abs/2403.10045) [`[Code]`](https://github.com/yumozi/GUARD)[`[Website]`](https://yumozi.github.io/GUARD/)

## Abstract
Dataset distillation (DD) allows datasets to be distilled to
fractions of their original size while preserving the rich distributional information so that models trained on the distilled
datasets can achieve a comparable accuracy while saving significant computational loads. Recent research in this area has
been focusing on improving the accuracy of models trained
on distilled datasets. In this paper, we aim to explore a new
perspective of DD. We study how to embed adversarial robustness in distilled datasets, so that models trained on these
datasets maintain the high accuracy and meanwhile acquire
better adversarial robustness. We propose a new method that
achieves this goal by incorporating curvature regularization
into the distillation process with much less computational
overhead than standard adversarial training. Extensive empirical experiments suggest that our method not only outperforms standard adversarial training on both accuracy and robustness with less computation overhead but is also capable
of generating robust distilled datasets that can withstand various adversarial attacks.

## Run all

Before running the code, you need to modify the Pytorch source code according to this document: [train/README.md](train/README.md).
  - -p: whether to train a new teacher model
  - -C: whether to use GUARD
  - -b: the batchnorm statistics regularization coefficient

```bash
bash run.sh -x 1 -y 1 -d imagenette -r /home/user/data/ -u 0 -b 10.0 -p -C -h 3.0 -l 100 >> output.log 2>&1 &
````

## Citation

```
@inproceedings{xue2025towards,
	author = {Eric Xue and Yijiang Li and Haoyang Liu and Peiran Wang and Yifan Shen and Haohan Wang},
	title = {Towards Adversarially Robust Dataset Distillation by Curvature Regularization},
	booktitle = {Proceedings of the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)},
	year = {2025},
}
```

## Acknowledgement

Our implementation is based on the code of [SRe<sup>2</sup>L](https://github.com/VILA-Lab/SRe2L).
