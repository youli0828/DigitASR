import subprocess
import sys


def test_default_pipeline_does_not_require_torch():
    code = '''
import importlib.abc
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch blocked for offline test", name="torch")

sys.meta_path.insert(0, NoTorch())
from digit_asr.pipeline import DigitRecognizer
from digit_asr.models import create_model
create_model("svm_hamming")
assert "torch" not in sys.modules
try:
    create_model("cnn")
except ImportError as error:
    assert ".[torch]" in str(error)
else:
    raise AssertionError("CNN must explain its optional dependency")
'''
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
