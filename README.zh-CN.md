<h1 align="center">AMMFV</h1>
<p align="center"><b>Adaptive Multi-modal Fact Verification via Expert Routing and Multi-Agent Debate</b></p>
<p align="center">
  <a href="README.md">English</a> ·
  <a href="#方法概览">方法</a> ·
  <a href="#训练">训练</a> ·
  <a href="#评测">评测</a> ·
  <a href="docs/usage.md">详细文档</a>
</p>

AMMFV 将自适应证据获取、异构图推理和多代理辩论用于多模态事实核验，输出事实判断及自然语言解释。本仓库提供 **MOCHEG** 与 **COSMOS** 上的数据准备、训练、推理和实验复现流程。

## 方法概览

<p align="center">
  <img src="assets/framework.png" alt="论文中的 AMMFV 框架图" width="100%">
</p>

框架包含三个模块：**自适应专家路由**选择直接生成、单步检索或多跳检索；**异构图结构学习**建模并筛选证据关系；**多代理辩论**通过独立证据判断、标准/反事实批评、Judge 裁决与 Rewriter 生成最终解释。

## 安装

使用 Python 3.10+，并安装适合本机 CUDA 环境的 PyTorch。从仓库根目录执行：

```bash
conda create -n ammfv python=3.10 -y
conda activate ammfv
python -m pip install -e .
```

FAISS、TensorBoard 和量化依赖可通过 `pip install -e ".[index,logging,quantization]"` 安装。

## 数据准备

从 [MOCHEG](https://github.com/PLUM-Lab/Mocheg) 与 [COSMOS](https://github.com/shivangi-aneja/COSMOS) 下载原始数据，保留标注、图片及目录结构，分别放入 `data/raw/mocheg/` 与 `data/raw/cosmos/`。

```bash
python -m ammfv prepare-dataset --config configs/datasets/mocheg.yaml --root data/raw/mocheg
python -m ammfv prepare-dataset --config configs/datasets/cosmos.yaml --root data/raw/cosmos
```

转换后，各数据集目录包含 `train.jsonl`、`validation.jsonl`、`test.jsonl`、`corpus.jsonl` 和准备记录。MOCHEG 保留三分类标签及参考解释；COSMOS 使用双 caption 上下文一致性任务，训练/验证对来自原始 caption 集合，测试使用原始标注。转换保留源数据划分并解析图片路径。

原始文件布局、标签映射和自定义数据格式见 [数据协议](docs/data_protocol.md)。

## 训练

默认使用 CLIP ViT-L/14 共享表示、BGE-large-en-v1.5 文本检索、LLaVA-v1.6-Mistral-7B 生成式代理和 DeBERTa-v3-base 路由器。先下载预训练模型：

```bash
python scripts/download_models.py --config configs/datasets/mocheg.yaml
```

### 阶段一：图结构学习

构建检索索引、缓存冻结 CLIP 特征，再以监督交叉熵训练图模块：

```bash
python -m ammfv build-index --config configs/datasets/mocheg.yaml
python -m ammfv cache-features --config configs/datasets/mocheg.yaml
python -m ammfv train-graph --config configs/datasets/mocheg.yaml
```

训练采用 AdamW 与 Gumbel 温度退火，验证集 macro-F1 选择 `checkpoints/mocheg/graph/best.pt`。

### 阶段二：专家路由器训练

在训练集上运行三类专家生成策略监督，然后训练只读取 claim 的路由器：

```bash
python -m ammfv router-labels --config configs/datasets/mocheg.yaml --output data/mocheg/router_labels.jsonl
python -m ammfv train-router --config configs/datasets/mocheg.yaml --labels data/mocheg/router_labels.jsonl
```

路由标签按成功的最简策略选择 S、M，否则 C。权重保存至 `checkpoints/mocheg/router/best/`。COSMOS 训练将配置及数据路径中的 `mocheg` 换为 `cosmos`，权重写入对应目录。

完整训练序列也可统一启动：

```bash
python scripts/run_pipeline.py --config configs/datasets/mocheg.yaml --stages index features graph router_labels router --execute
```

断点恢复、多 GPU、检索证据缓存和模型加载见 [详细使用文档](docs/usage.md#training)。

## 评测

训练完成后，对测试集执行自适应获取及核验：

```bash
python -m ammfv infer --config configs/datasets/mocheg.yaml
python -m ammfv evaluate --config configs/datasets/mocheg.yaml --predictions outputs/mocheg/retrieved/full/predictions.jsonl --output outputs/mocheg/retrieved/full/evaluation.json
```

COSMOS 将命令中的 `mocheg` 换为 `cosmos`。Gold evidence 使用对应实验配置：

```bash
python -m ammfv infer --config configs/experiments/mocheg_gold.yaml
python -m ammfv evaluate --config configs/experiments/mocheg_gold.yaml --predictions outputs/mocheg/gold/full/predictions.jsonl --output outputs/mocheg/gold/full/evaluation.json
```

评测输出 **Accuracy、Precision、Recall、F1**，并对参考解释计算 **ROUGE-1/2/L、BLEU-1..4**。运行目录保存预测、指标及完整证据/代理 trace。自定义 claim、固定专家与分片推理见 [使用文档](docs/usage.md#evaluation-and-inference)。

## 实验复现

主矩阵覆盖两个数据集、retrieved/gold 两种证据协议，以及去图像、固定检索、去图和去辩论四项消融：

```bash
python scripts/launch.py --spec configs/experiments/matrix.yaml --execute
```

Reference 数量与独立证据代理数量的敏感性实验：

```bash
python scripts/launch.py --spec configs/sweeps/reference_size.yaml --execute
python scripts/launch.py --spec configs/sweeps/agent_count.yaml --execute
```

配置位于 `configs/{datasets,models,hardware,ablations,sweeps}/`，使用 `--set key=value` 覆盖单项参数。实验协议及冲突组配对分析见 [复现说明](docs/reproduction.md)；baseline、可视化和结果汇总见 [使用文档](docs/usage.md)。

## 项目结构

```text
AMMFV/
├── assets/          # 论文框架图
├── configs/         # 数据集、模型、硬件及实验配置
├── src/ammfv/       # 检索、图、代理、训练及评测
├── scripts/         # 数据处理、实验启动及结果分析
├── schemas/         # 数据与预测格式
├── docs/            # 详细使用与方法文档
└── results/         # 论文参考表格
```

更多实现细节见 [论文—代码对应](docs/paper_to_code.md)、[API](docs/api.md) 和 [完整目录](PROJECT_TREE.md)。
