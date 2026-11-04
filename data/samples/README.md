# 中文孤立数字合成样例

此目录存放 120 条原始生成的普通话数字 WAV（零至九，每类 12 条），
16 kHz、单声道、PCM16。没有下载真人录音，也没有用正弦波冒充语音。
引擎为 [eSpeak NG](https://github.com/espeak-ng/espeak-ng)，使用 `cmn` 普通话
共振峰合成音色；通过 [espeakng-loader 0.2.4](https://pypi.org/project/espeakng-loader/0.2.4/)
加载。确切引擎版本和合成参数见 `provenance.json`。

`manifest.csv` 逐条记录相对路径、数字标签、划分、来源组、原文、语速、音高和 SHA-256。
固定 12 个 `(rate, pitch)` 参数组合，每组包含完整十个数字，其中 8 组训练（80 条），
4 组测试（40 条）。测试参数组合在生成前已固定，训练没有读取测试特征或调参。
各条语音直接合成，不从同一基础 WAV 加噪或变速后随机拆分。

所有样本仍共享同一合成器和音色，所以这是**合成域内的流程验证**，不是跨说话人或真人
测试；参数组不代表不同说话人。不能据此宣称真人准确率达到 90%。数据很小，单条预测
就影响总体准确率 2.5 个百分点。

生成命令（从项目根目录）：

```powershell
python -m pip install -e ".[data]"
python scripts/generate_samples.py --output data/regenerated
```

新目录可用于重新生成；脚本拒绝覆盖已存在的数据清单。日常训练直接使用随仓库提供的 WAV，
无需安装或调用合成引擎。不同平台的合成器/数值库可能产生不同字节，复现实测请使用已有 WAV。

本项目生成的数字文本、清单和 WAV 以项目根目录 MIT LICENSE 提供。
eSpeak NG 引擎本身遵循 GPL-3.0，未将引擎二进制复制到本目录；它是可选生成工具。
本语料使用共振峰计算合成，不使用 MBROLA 或外部音频片段。
关于程序输出与程序许可证的一般区别，参见
[GNU GPL FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#WhatCaseIsOutputGPL)。
