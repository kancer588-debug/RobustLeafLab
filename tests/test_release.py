from contextlib import redirect_stdout
import csv
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PIL import Image, ImageOps
import torch
from torch import nn

from robustleaflab.__main__ import main
from robustleaflab.directional import DirectionalReview, EMBED_DIM
from robustleaflab.models import PREPROCESS, read_image
from robustleaflab.replay import evaluate


class StubClassifier:
    name = "current"
    model_version = "v0.2.0"

    def __call__(self, images):
        return torch.tensor([[2.0, 1.0, 0.0]]).repeat(len(images), 1)


class FixedHead(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = torch.tensor(scores)

    def forward(self, *args):
        return self.scores


class InputAndReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.first = self.root / "first.png"
        self.last = self.root / "last.png"
        for path in (self.first, self.last):
            Image.new("RGB", (224, 224), "green").save(path)

    def test_prediction_applies_exif_when_requested(self):
        image = Image.new("RGB", (224, 302), "black")
        image.paste("white", (0, 0, 112, 120))
        orientation = Image.Exif()
        orientation[274] = 6
        tagged = self.root / "rotated.jpg"
        image.save(tagged, exif=orientation)
        with Image.open(tagged) as raw:
            expected = PREPROCESS(ImageOps.exif_transpose(raw).convert("RGB"))
        self.assertTrue(torch.equal(read_image(tagged, apply_exif=True), expected))
        self.assertFalse(torch.equal(read_image(tagged), expected))

    def test_bad_image_emits_an_error_and_keeps_processing(self):
        broken = self.root / "broken.jpg"
        broken.write_text("invalid image", encoding="utf-8")
        with patch("robustleaflab.__main__.RobustLeafClassifier", return_value=StubClassifier()):
            output = StringIO()
            with redirect_stdout(output):
                code = main(["predict", str(self.first), str(broken), str(self.last), "--device", "cpu"])
        self.assertEqual(code, 1)
        results = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([item["image"] for item in results], list(map(str, [self.first, broken, self.last])))
        self.assertEqual([item.get("label") for item in results], ["healthy", None, "healthy"])
        self.assertIn("error", results[1])
        self.assertTrue(all(item["model_version"] == "v0.2.0" for item in results))

    def write_manifest(self, expected, digest):
        manifest = self.root / "manifest.csv"
        with manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "sha256", "source", "true_label", "expected_current"])
            writer.writerow([self.first.name, digest, "test", "early_blight", expected])
        return manifest

    def test_replay_checks_hash_collects_matrix_and_reports_frozen_mismatch(self):
        digest = sha256(self.first.read_bytes()).hexdigest()
        manifest = self.write_manifest("early_blight", digest)
        report = evaluate(manifest, None, StubClassifier(), check_frozen=True)
        self.assertEqual(report["images"], 1)
        self.assertEqual(report["correct"], 0)
        self.assertEqual(report["confusion_matrix"], [[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(report["frozen_mismatches"], 1)
        with patch("robustleaflab.__main__.RobustLeafClassifier", return_value=StubClassifier()):
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["evaluate", str(manifest), "--check-frozen", "--device", "cpu"]), 1)
        invalid = self.write_manifest("healthy", "0" * 64)
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            evaluate(invalid, None, StubClassifier(), check_frozen=True)


class VoteTests(unittest.TestCase):
    def test_invalid_seed_sets_fail_before_loading_weights(self):
        with TemporaryDirectory() as location:
            folder = Path(location)
            with patch("robustleaflab.directional.HEADS_DIR", folder):
                for seeds in ([42, 45], [42, 42, 46], [42, 45, 46, 47]):
                    (folder / "manifest.json").write_text(
                        json.dumps({"heads": [{"seed": seed} for seed in seeds]}), encoding="utf-8")
                    with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "exactly one head pair"):
                        DirectionalReview()

    def test_two_votes_switch_either_subtype_and_rescue_original_healthy(self):
        review = DirectionalReview()
        review.early_rescue = nn.ModuleList([
            FixedHead([0., 0., 5., 0.]), FixedHead([0., 0., 5., 0.]), FixedHead([0., 0., 0., 0.]),
        ])
        review.subtype = nn.ModuleList([
            FixedHead([4., -4., -4., -4.]), FixedHead([4., 0., -4., -4.]), FixedHead([0., 0., -4., 0.]),
        ])
        scores = torch.tensor([[0., .2, .4], [0., .4, .2], [.4, .3, .2], [0., .4, .2]])
        with torch.inference_mode():
            updated = review(scores, torch.zeros(4, EMBED_DIM), torch.zeros(4, 4, EMBED_DIM),
                             torch.zeros(4, 2 * EMBED_DIM))
        self.assertEqual(updated.argmax(1).tolist(), [1, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
