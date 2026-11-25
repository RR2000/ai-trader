"""Regenerates tests/fixtures/keras_reference.npz with TensorFlow's Keras (not needed by the tests).

    pip install tensorflow==2.21.0 h5py && python tests/make_keras_reference.py

The inputs are random windows plus real-looking ones (built from a random walk with
trader.features), the outputs are what Keras predicts for each saved model.
"""

import json
import sys
from pathlib import Path

import h5py

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402

from trader import features  # noqa: E402

rng = np.random.default_rng(2023)
random_windows = rng.uniform(0.01, 1.01, size=(24, 30, 7))
real = []
for _ in range(8):
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.0008, 400)))
    volume = rng.gamma(2.0, 5.0, 400)
    real.append(features.observation(close, volume))
inputs = np.concatenate([random_windows, np.stack(real)]).astype(np.float32)

models = sorted(p.name for p in (ROOT / "src/model/completed").glob("*.h5"))
out = {"inputs": inputs, "models": np.array(models), "keras_version": np.array(keras.__version__),
       "backend": np.array(keras.backend.backend())}
for name in models:
    path = ROOT / "src/model/completed" / name
    # the architecture saved in the file itself; Keras 3 only refuses LSTM's old time_major argument
    with h5py.File(path, "r") as f:
        config = json.loads(f.attrs["model_config"])
    for layer in config["config"]["layers"]:
        layer["config"].pop("time_major", None)
    model = keras.Sequential.from_config(config["config"])
    model.load_weights(str(path))
    out[f"q_{name}"] = model.predict(inputs, verbose=0)
(ROOT / "tests/fixtures").mkdir(exist_ok=True)
np.savez(ROOT / "tests/fixtures/keras_reference.npz", **out)
print("saved", {k: v.shape for k, v in out.items()}, keras.__version__, keras.backend.backend())
