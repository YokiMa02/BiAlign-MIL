"""Checks runnable without PyTorch; this is not a model runtime test."""
import ast
import hashlib
from pathlib import Path
import pickle
import tempfile

import numpy as np
from data import load_pair


def main():
    root = Path(__file__).resolve().parent
    for path in root.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    source = root / "models/model_bialign_ms.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "8a5a240f17ebc9f155975666b2fc5fe8d6886f387e56613e3e3f16ab165b7a67"
    with tempfile.TemporaryDirectory() as tmp:
        h, l = Path(tmp)/"high.pkl", Path(tmp)/"low.pkl"
        high = [dict(coord="27_137", low_coord="13_68", feature=np.full(768, 7)),
                dict(coord="30_30", low_coord="15_15", feature=np.full(768, 8)),
                dict(coord="27_136", low_coord="13_68", feature=np.full(768, 9))]
        low = [dict(coord="13_68", feature=np.full(768, 3))]
        h.write_bytes(pickle.dumps(high)); l.write_bytes(pickle.dumps(low))
        x, c = load_pair(h, l)
        assert x.shape == (3, 1536) and c.shape == (3, 5)
        np.testing.assert_array_equal(x[:, 0], [7, 8, 9])
        np.testing.assert_array_equal(x[:, 768], [3, 0, 3])
        np.testing.assert_array_equal(c, [[27,137,13,68,1],[30,30,0,0,0],[27,136,13,68,1]])
        l.write_bytes(pickle.dumps([]))
        x, c = load_pair(h, l)
        assert not x[:,768:].any() and not c[:,4].any()
        l.write_bytes(pickle.dumps(low + low))
        try:
            load_pair(h, l)
        except ValueError:
            pass
        else:
            raise AssertionError("Duplicate parent coordinates must be rejected")
    print("PASS: Python syntax, original V1 SHA256, pairing/order/missing-Low checks")
    print("NOT TESTED: model forward/backward, checkpoint loading, GPU training")


if __name__ == "__main__":
    main()
