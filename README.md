# myTextFooler

A Model for Natural Language Attack on Text Classification and Inference

使用时序评分对单词重要性排序，对关键词做同意替换，选取替换前后相似度较大且影响较大样本作为对抗样本。

### 环境

本仓库仿照[TextFooler](https://github.com/jind11/TextFooler)，对平台进行了适配。相关环境安装请参考原仓库，或者参考`build.sh`。

### 运行

在环境安装好后，请在`myTextFooler`目录下解压`counter-fitted-vectors.txt.zip`

先构建TextFooler类的实例：

```python
textfooler = TextFooler(model=predictor, device='gpu0', IsTargeted=False)
```

然后将输入数据与标签输入，来生成对抗样本：

```python
adv_xs = textfooler.generate(texts, labels)
```

具体运行样例参考`text.py`（注：需要原仓库中的一些代码依赖，具体看import。`text.py`仅为测试，具体使用不需要原仓库依赖）

本仓库代码仅在与原仓库相同配置下测试过。

### 参数说明

```
perturb_ratio: Whether use random perturbation for ablation study. 默认为0

sim_score_threshold: Required minimum semantic similarity score. 默认为0.7

import_score_threshold: Required mininum importance score. 默认为-1

sim_score_window: Text length or token number to compute the semantic similarity score  默认为15

synonym_num: Number of synonyms to extract  默认为50

batch_size: Batch size to get prediction  默认为32

counter_fitting_embeddings_path: path to the counter-fitting embeddings we used to find synonyms  默认为myTextFooler/counter-fitted-vectors.txt

counter_fitting_cos_sim_path: pre-compute the cosine similarity scores based on the counter-fitting embeddings  默认为空

USE_model_path: Path to the USE model.  默认为空

```

