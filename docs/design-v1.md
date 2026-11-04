# 开源项目设计方案 v1（浩民）

> 主导：@项目　实现：@codex　简历落地：@简历　协调：@小羊  
> 依据：杨浩民简历（自动化 / C·Python·信号处理）+ 李政翰简历写法（独立开源 + GitHub 链接 + 可量化成果）

---

## 1. 推荐方向（主方案）

**项目名**：DigitASR  
**一句话**：面向 0–9 中文孤立数字的端到端语音识别开源工具箱，含预处理、特征提取、多模型对比与可交互 Demo。

**选型理由**
- 与浩民已有「孤立数字语音识别」经历强对齐（曾做到约 90%），开源后可从「核心成员」升级为「独立主导开源项目」
- 技术栈贴合专业：信号处理 + Python + 经典 ML（可选轻量深度学习）
- 体量可控，便于 @codex 在 GitHub 快速落地；简历写法可对标同学（开源地址 + 指标 + 模块职责）

**备选（若更想「产品感」）**  
BreathViz：可穿戴呼吸监测数据可视化 + 简单训练反馈 Web Demo（来自科创项目）。实现稍重、传感器数据需模拟；本版默认推进 DigitASR。

---

## 2. 目标用户与价值

| 对象 | 价值 |
|------|------|
| 课程/自学 | 可复现的中文数字 ASR 教学流水线 |
| 简历审阅者 | 清晰的个人开源作品：结构完整、可跑 Demo、有指标 |
| 二次开发者 | 模块化接口，可替换特征/模型 |

**成功标准（MVP）**
1. `pip install -e .` 后本地可跑通识别与评估  
2. 提供 Gradio/Streamlit Demo（上传 wav 或录音 → 输出数字）  
3. README 含原理简述、快速开始、结果表、开源协议  
4. 在公开小样本/自录样本上给出可复现准确率（目标：单人受控语料 ≥ 90%，与已有经历一致）  
5. GitHub Actions：lint + pytest 通过  

---

## 3. 功能范围

### MVP（必须）
- 音频输入：wav 文件；可选麦克风（Demo 内）
- 预处理：预加重、分帧、加窗（对比至少 2 种窗，呼应已有经历）
- 特征：MFCC（主）+ 可选时域特征基线
- 模型：至少 2 种（如 SVM / 随机森林；或 GMM + SVM）可切换对比
- 评估：混淆矩阵、准确率、按数字分类报告；导出简单结果表
- Demo：Web 交互一页
- 文档：中文 README 为主，英文 Brief 可选

### 非目标（首版不做）
- 连续语音 / 大规模 ASR
- 移动端 App、硬件采集板驱动
- 完整自研大数据集托管（仅放少量 sample + 录制说明）

### 增强（有余力）
- 轻量 CNN/TDNN（PyTorch）对比经典模型
- 简单数据增强（加噪、变速）
- 导出 ONNX

---

## 4. 技术栈

| 层 | 选型 |
|----|------|
| 语言 | Python 3.10+ |
| 数值/音频 | NumPy, SciPy, librosa, soundfile |
| 模型 | scikit-learn（MVP）；可选 PyTorch（增强） |
| Demo | Gradio（优先，部署简单）或 Streamlit |
| 工程 | pyproject.toml, pytest, ruff, GitHub Actions |
| 协议 | MIT |

---

## 5. 仓库结构

```text
DigitASR/
├── README.md
├── LICENSE
├── pyproject.toml
├── .github/workflows/ci.yml
├── src/digit_asr/
│   ├── __init__.py
│   ├── preprocess.py      # 预加重、分帧、加窗
│   ├── features.py        # MFCC 等
│   ├── models.py          # 训练/加载/预测接口
│   ├── evaluate.py
│   └── pipeline.py        # wav → 数字
├── demos/
│   └── app.py             # Gradio Demo
├── scripts/
│   ├── train.py
│   ├── eval.py
│   └── record_guide.md    # 自录数据说明
├── tests/
├── data/samples/          # 少量示例 wav（0–9 各数条）
└── docs/
    └── design-notes.md
```

---

## 6. README 大纲

1. 徽章（CI / License / Python）  
2. 项目简介 + Demo 截图占位  
3. 特性列表  
4. 快速开始（安装 / 识别一条 / 启动 Demo）  
5. 流水线说明（预处理 → 特征 → 模型）  
6. 实验结果（表格 + 简要分析，含窗函数对比）  
7. 目录结构  
8. 如何贡献 / 录制自己的数据  
9. 致谢与 License  

---

## 7. 里程碑（建议交给 @codex）

| 阶段 | 内容 | 产出 |
|------|------|------|
| M1 | 仓库骨架、包结构、预处理 + MFCC、单元测试 | 可 import 的包 |
| M2 | 训练/评估脚本、≥2 模型、sample 数据跑通 | 准确率报告 |
| M3 | Gradio Demo、完善 README、CI、MIT | 可展示的开源仓库 |
| M4* | 可选：轻量神经网络或数据增强 | 对比实验写入 README |

\*M4 非阻塞简历写入；M3 完成后即可给 @简历。

---

## 8. 简历可写要点（供 @简历 预览）

**DigitASR｜中文孤立数字语音识别开源工具箱｜独立开发**  
- 设计并开源完整识别流水线（预加重 / 分帧加窗 / MFCC / 多模型对比），提供 Gradio 交互 Demo 与可复现评估脚本  
- 对比多种窗函数与分类器，在单人受控语料上达到约 XX% 准确率（以仓库实测为准）  
- 工程化：Python 包结构、pytest、GitHub Actions CI；开源地址：`https://github.com/<浩民用户名>/DigitASR`

---

## 9. 待确认（请 @小羊 / 浩民）

1. **GitHub 用户名 / 是否已有空仓库**（@codex 推送需要）  
2. **主方案 DigitASR 是否拍板**；若更想 BreathViz 再说一声我改一版设计  
3. 是否需要英文 README（默认中文为主 + 英文 Brief）

---

## 10. 下一步

- 拍板后 → @codex 按本方案建仓与实现（从 M1 起）  
- M3 有可访问仓库后 → @简历 写入项目并按李政翰简历调整版式  
