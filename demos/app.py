import argparse
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr

from digit_asr.pipeline import DigitRecognizer

ROOT = Path(__file__).resolve().parents[1]
CHOICES = [
    f"{model}_{window}" for model in ("svm", "random_forest") for window in ("hamming", "hann")
]


@lru_cache(maxsize=4)
def get_recognizer(name):
    path = ROOT / "artifacts" / name / "model.joblib"
    if not path.exists():
        raise gr.Error("请先按 README 运行对应模型的训练命令。")
    return DigitRecognizer(path)


def recognize(audio, model):
    if not audio:
        raise gr.Error("请上传 WAV 或录制一个中文数字。")
    if model not in CHOICES:
        raise gr.Error("请选择列表中的模型。")
    try:
        digit = get_recognizer(model).predict(audio)
    except (ValueError, RuntimeError) as error:
        raise gr.Error(f"无法读取有效语音：{error}") from error
    return str(digit)


def build_app():
    examples = sorted((ROOT / "data/samples/test").glob("*.wav"))[:10]
    with gr.Blocks(title="DigitASR 中文孤立数字识别") as app:
        gr.Markdown(
            "# DigitASR\n上传或录制**一个**普通话数字（零至九）。\n\n"
            "示例模型仅由 eSpeak NG 合成语音训练，真人录音效果尚未验证。"
        )
        audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="数字语音")
        model = gr.Dropdown(CHOICES, value="svm_hamming", label="模型 / 窗函数")
        output = gr.Textbox(label="识别数字")
        gr.Button("识别", variant="primary").click(recognize, [audio, model], output)
        gr.Examples([[str(path)] for path in examples], inputs=[audio])
    return app


if __name__ == "__main__":
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key), "127.0.0.1", "localhost"]))
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().launch(server_name="127.0.0.1", server_port=args.port, share=False)
