# LeRobot ACT × CALVIN 跨环境泛化

基于 [LeRobot](https://github.com/huggingface/lerobot) 内置 **ACT(Action Chunking with Transformers)** 算法,在 CALVIN 数据集上研究视觉-动作策略的**跨环境泛化**能力:

1. **单环境训练**:仅用环境 A 数据训练 `ACT-A`
2. **多环境联合训练**:用 A+B+C 混合数据,以**完全相同的架构与超参**训练 `ACT-ABC`
3. **Zero-shot 跨环境测试**:在未见过的环境 D 上对比两模型的离线动作误差(L1/MSE),分析 Action Chunking 在视觉分布偏移下的鲁棒性



---

## 1. 环境配置

本项目需要 **Python 3.10 + CUDA 11.8** 的 GPU 环境(实测 GTX 1660 Ti 6GB 可跑)。

```bash
# 1. 建 conda 环境
conda create -n lerobot-act python=3.10 -y
conda activate lerobot-act

# 2. 装 GPU 版 torch(cu118)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# 3. 装 lerobot 与本项目依赖
pip install lerobot==0.4.4
pip install -r requirements.txt

# 4. 引入本项目包(中文路径下用 PYTHONPATH,不要 pip install -e .)
export PYTHONPATH=src      # Windows PowerShell: $env:PYTHONPATH="src"
```

> Windows 控制台若中文乱码,命令前加 `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`。

---

## 2. 数据准备

数据集 [`huiwon/calvin_task_ABC_D`](https://huggingface.co/datasets/huiwon/calvin_task_ABC_D)(LeRobot v2.1 格式,约 7.8GB,4 个分片)。

```bash
# 1. 下载(带断点续传、官方/镜像自动切换、并发)
python 下载脚本.py              # 下全部 4 分片
python 下载脚本.py 0            # 只下分片0(先跑通)

# 2. 校验完整性(文件数量)
python -m scripts.verify_data

# 3. 校验视频可解码(揪出下载损坏的 mp4)
python -m scripts.check_videos
python 下载脚本.py --from-list outputs/bad_videos.txt   # 重下损坏文件

# 4. 转换格式 v2.1 → v3.0(lerobot 0.4.x 必需)
python -m scripts.convert_to_v30        # 转全部分片

# 5. 生成 A / ABC / D 数据视图(ABC 为 A+B+C 合并)
python -m scripts.make_splits --envs A ABC D
```

**环境↔分片映射**(定义于 `configs/act_calvin.yaml`):

| 分片目录 | 环境 | 用途 |
|---|---|---|
| `..._0_4` | A | 任务1训练 / 任务2训练之一 |
| `..._1_4` | B | 任务2训练之一 |
| `..._2_4` | C | 任务2训练之一 |
| `..._3_4` | D | 任务3 Zero-shot 测试(不进训练) |

---

## 3. 训练

任务1与任务2**共用同一脚本与同一份超参**(`configs/act_calvin.yaml`),仅 `--env` 不同,从机制上保证"同架构同超参"。

```bash
# 任务1:仅环境 A
python -m scripts.train --env A --steps 50000

# 任务2:A+B+C 联合(config 完全一致,只换数据)
python -m scripts.train --env ABC --steps 50000
```

权重输出到 `outputs/act_A/` 与 `outputs/act_ABC/`。开 WandB 记录加 `--wandb`。

---

## 4. Zero-shot 评估(环境 D)

```bash
# 分别评估两模型在 D 上的离线动作误差
python -m scripts.eval_zeroshot \
  --ckpt outputs/act_A/checkpoints/<step>/pretrained_model --test-env D
python -m scripts.eval_zeroshot \
  --ckpt outputs/act_ABC/checkpoints/<step>/pretrained_model --test-env D

# 管线冒烟(只跑2条 episode)
python -m scripts.eval_zeroshot --ckpt <...> --test-env D --limit 2
```

输出 L1/MSE 与**块内逐步误差**到 `outputs/eval/*.json`,供报告出图。

---

## 5. 项目结构

```
├── configs/act_calvin.yaml      # 单一真相源:环境映射 + ACT 超参
├── src/calvin_act/
│   ├── config.py                # 配置加载与校验(断言同超参、D 不泄漏)
│   ├── data/{splits,merge,verify}.py
│   ├── eval/{metrics,runner,visualize}.py
│   └── utils/{io,seed}.py
├── scripts/                     # 薄入口:verify_data/check_videos/convert_to_v30/
│                                #   make_splits/train/eval_zeroshot
├── 下载脚本.py                   # CALVIN 下载器
└── tests/                       # 纯函数单测(pytest)
```

---

## 6. 测试

```bash
python -m pytest tests/ -v       # 指标纯函数单测(不需 GPU)
```
