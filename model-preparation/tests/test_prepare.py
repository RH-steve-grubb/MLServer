"""Small regression tests for fixture generation and correctness checks."""

import copy
import json
from pathlib import Path
import socket
import tempfile
from typing import Any
import unittest

import numpy as np

from prepare import compare_detections, prepare_repositories, write_json
from validate import check_ports, decoded, grpc_request


class FixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.expected: dict[str, list[Any]] = {
            "xyxy": [[0, 0, 20, 20], [30, 30, 50, 50]],
            "confidence": [0.9, 0.8],
            "class_id": [17, 90],
        }

    def test_detection_parity_accepts_reordering(self) -> None:
        actual = {key: list(reversed(value)) for key, value in self.expected.items()}
        compare_detections(self.expected, actual)

    def test_detection_parity_rejects_invalid_results(self) -> None:
        for field, replacement in (
            ("class_id", [17, 17]),
            ("confidence", [0.1, 0.8]),
            ("xyxy", [[100, 100, 120, 120], [30, 30, 50, 50]]),
        ):
            with self.subTest(field=field):
                actual = copy.deepcopy(self.expected)
                actual[field] = replacement
                with self.assertRaises(ValueError):
                    compare_detections(self.expected, actual)

    def test_coco_final_class_is_not_discarded(self) -> None:
        logits = np.full((1, 1, 91), -20, dtype=np.float32)
        logits[0, 0, 90] = 10
        boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
        result = decoded({"dets": boxes, "labels": logits}, [100, 50])
        self.assertEqual(result["class_id"], [90])
        np.testing.assert_allclose(result["xyxy"], [[25, 12.5, 75, 37.5]])

    def test_repositories_are_portable_and_batching_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepare_repositories(root)
            for device in ("cpu", "cuda"):
                path = root / "repositories" / device / "rfdetr-small"
                settings = json.loads((path / "model-settings.json").read_text())
                uri = (path / settings["parameters"]["uri"]).resolve()
                self.assertEqual(uri, root / "model" / "model.onnx")
                self.assertEqual(settings["max_batch_size"], 0)
                providers = settings["parameters"]["extra"]["providers"]
                self.assertEqual("CUDAExecutionProvider" in providers, device == "cuda")

    def test_raw_grpc_fixture_requests_both_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "sample").mkdir()
            tensor = np.zeros((1, 3, 512, 512), dtype="<f4")
            tensor.tofile(root / "sample" / "input.fp32")
            request = grpc_request(root)
            self.assertEqual(
                [value.name for value in request.outputs], ["dets", "labels"]
            )
            self.assertEqual(list(request.inputs[0].shape), list(tensor.shape))
            self.assertEqual(len(request.raw_input_contents[0]), tensor.nbytes)
            self.assertEqual(request.raw_input_contents[0], tensor.tobytes())

    def test_compact_json_preserves_float32_values(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.json"
            tensor = np.array([0.1, -2.3], dtype=np.float32)
            write_json(path, {"data": tensor.tolist()}, compact=True)
            result = np.array(json.loads(path.read_text())["data"], dtype=np.float32)
            np.testing.assert_array_equal(result, tensor)
            self.assertNotIn("\n ", path.read_text())

    def test_port_probe_rejects_an_existing_listener(self) -> None:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(OSError):
                check_ports((listener.getsockname()[1],))

    def test_port_probe_accepts_recently_closed_connections(self) -> None:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            address = listener.getsockname()
            with socket.create_connection(address, timeout=2) as client:
                peer, _ = listener.accept()
                with peer:
                    peer.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
        check_ports((address[1],))


if __name__ == "__main__":
    unittest.main()
