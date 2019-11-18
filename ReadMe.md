# Unweighted Influence Data Subsampling (UIDS)

This repository provides a numpy and scipy based implementation on *Unweighted Influence Data Subsampling* (UIDS) on the Logistic Regression model. The UIDS can achieve good result when the data set quality is not good, such as noisy labels, or there is distribution shift between training and test set, by dropping several bad cases.

## Paper & Citation

**Less Is Better: Unweighted Data Subsampling via Influence Function**

Zifeng Wang 1, Hong Zhu 2, Zhenhua Dong 2, Xiuqiang He 2, Shao-Lun Huang 1

1 Tsinghua-Berkeley Shenzhen Institute, 2 Noah's Ark Lab, Huawei

*34th AAAI Conference on Artificial Intelligence (AAAI)*, 2020



------

If you find this work interesting or helpful for your research, please consider citing this paper and give your star ^ ^



````latex
@inproceedings{wang2020influence,
  title={Less Is Better: Unweighted Data Subsampling via Influence Function},
  author={Wang, Zifeng and Zhu, Hong and Dong, Zhenhua and He, Xiuqiang and Huang, Shao-Lun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={1},
  pages={1--8},
  year={2020}
}
````



## Introduction

In practice, it is common that some of the collected data are not good or even harmful for our model. On the other, sometimes the data distribution is not stable. There often lies distribution shift between the training set and the test set, causing the degradation performance of the classical supervised machine learning algorithms.

### Intuition Demonstration

Subsampling tries to build a tool to quantify each data’s quality, thereby keeping good examples and dropping bad examples to improve model’s generalization ability. Previous works concentrate on *Weighted* subsampling, that is, trying to maintain the model performance when dropping several data. 

By contrast, our work attempts to obtain a superior model by subsampling. 

The different between them can be shown as the image below:

<br/>
<p align="left"><img width="100%" src="figure/fig1.png" /></p>

-  **(a)** means if the blue points (training samples) within the red circle are removed, the new optimal decision boundary is still same as the former one

- **(b)** if removing blue points in the red circle, the new decision boundary shifts from the left, while achieves better performance on the Te set
<br/>

### Main Framework
<br/>
The main process of doing subsampling is as follows:

- **(a)** first train a model on the full data set

- **(b)** compute the *influence function* (IF) for each sample in training set

- **(c)** compute the sampling probability of each sample in training set 

- **(d)** doing subsampling and train a subset-model and the reduced data set

  

<p align="left"><img width="70%" src="figure/fig2.png" /></p>

<br/>



## Usage & Demo

### For simple Demo on MNIST and Breast-cancer

We have prepared simple demo on Logistic Regression and SVM, see in **Demo_on_Logistic_Regression.ipynb** and **Demo_on_SVM.ipynb**

The experiment results would be shown as following. We have the *Sig-UIDS* obtain ACC and AUC much better than the Full-set-model and subset-model obtained by random sampling.

```shell
============================================================
MNIST: Result Summary on Te (ACC and AUC)
[SigUIDS]  acc 0.984281, auc 0.998802 # 4994
[Random]   acc 0.900139, auc 0.950359 # 4995
[Full]     acc 0.916782, auc 0.961824 # 8325
============================================================
```

### For other data sets

#### TODO



## Acknowledgement

This work was mainly done while the first author Zifeng Wang did a research internship at [Noah's Ark Lab, Huawei](http://www.noahlab.com.hk/). We especially thank the insights and advice from Professor [Chih-Jen Lin](https://www.csie.ntu.edu.tw/~cjlin/) for the theory and writing of this work.