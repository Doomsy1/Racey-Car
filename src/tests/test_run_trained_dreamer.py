import argparse
import json
import os
import tempfile
import unittest

import run_trained_dreamer


class RunTrainedDreamerTests(unittest.TestCase):
    def test_apply_metadata_overrides_uses_trained_config_when_default_eval_config_is_passed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"config_path": "models/track_config.yaml", "obs_scale": 0.25}, f)
            args = argparse.Namespace(
                model=tmpdir,
                config=run_trained_dreamer._default_eval_config_path(),
                obs_scale=0.5,
            )

            run_trained_dreamer._apply_metadata_overrides(args)

            self.assertEqual(args.config, "models/track_config.yaml")
            self.assertEqual(args.obs_scale, 0.25)

    def test_apply_metadata_overrides_keeps_explicit_config_override(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump({"config_path": "models/track_config.yaml", "obs_scale": 0.25}, f)
            args = argparse.Namespace(
                model=tmpdir,
                config="models/track_config_eval_random.yaml",
                obs_scale=0.5,
            )

            run_trained_dreamer._apply_metadata_overrides(args)

            self.assertEqual(args.config, "models/track_config_eval_random.yaml")
            self.assertEqual(args.obs_scale, 0.25)


if __name__ == "__main__":
    unittest.main()
