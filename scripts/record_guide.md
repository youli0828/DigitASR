# 录制自己的语料

每个 WAV 只说一个普通话数字：零、一、二、三、四、五、六、七、八、九。
首版标签 1 对应“一”，不把“幺”作为单独类别。建议 16 kHz、单声道、PCM16，
每条约 0.2–2 秒，避免削波；加载器也支持其他采样率和立体声。

每个数字建议每人至少录制 20 条，并覆盖不同距离、语速、设备和背景噪声。
取得录音者同意后再发布；记录来源和授权，不上传包含个人信息的录音。

新建独立数据目录及 UTF-8 `manifest.csv`：

```csv
path,label,split,group
train/speaker01_0_01.wav,0,train,speaker01
test/speaker02_0_01.wav,0,test,speaker02
```

上例只展示字段，完整文件的 train/test 各自必须覆盖 0–9。`path` 相对于清单目录；
`group` 表示说话人或录音会话，同一组只能属于一个划分。要验证跨人泛化，应按说话人分组；
验证单人跨会话效果，可按会话分组。先划分原始录音，再仅对训练集增强，禁止把同一录音的
不同变体放入测试集。工具会拒绝跨集相同文件内容和跨集相同 group。

```powershell
python scripts/train.py --manifest data/my_digits/manifest.csv --model svm --output artifacts/my_svm/model.joblib
python scripts/eval.py --manifest data/my_digits/manifest.csv --model artifacts/my_svm/model.joblib
```

Demo 默认加载 `artifacts/{模型}_{窗函数}/model.joblib`，若要在 Demo 使用自己的数据，
训练时将 `--output` 指向对应位置。只加载可信来源的 joblib 文件。
