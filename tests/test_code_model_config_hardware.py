"""Tests for training/code_model/config.py and hardware.py that work
regardless of whether the heavy ML stack (torch/transformers/bitsandbytes)
is installed in the environment pytest runs under -- detect_hardware()
must degrade gracefully (never raise) when they're missing, exactly like
it must report real values when they're present. Full functional
GPU/bitsandbytes verification lives in the .venv-code-model smoke test
(scripts/code_model_smoke_test.py), not here.
"""
import tempfile
import unittest
from pathlib import Path

from training.code_model.config import CodeModelTrainingConfig, list_configs, load_config
from training.code_model.hardware import HardwareInfo, check_feasibility, detect_hardware


class ConfigTests(unittest.TestCase):
    def test_shipped_configs_load_without_error(self):
        for name in ("small_smoke_test", "qlora_7b", "qlora_14b"):
            with self.subTest(name=name):
                config = load_config(name)
                self.assertEqual(config.name, name)
                self.assertNotEqual(config.base_model.model_id, "not-configured")

    def test_list_configs_finds_all_three(self):
        names = list_configs()
        self.assertIn("small_smoke_test", names)
        self.assertIn("qlora_7b", names)
        self.assertIn("qlora_14b", names)

    def test_unknown_config_name_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_config("this-config-does-not-exist")

    def test_config_hash_is_stable_and_sensitive_to_changes(self):
        a = load_config("small_smoke_test")
        b = load_config("small_smoke_test")
        self.assertEqual(a.config_hash(), b.config_hash())
        b.lora.rank = 999
        self.assertNotEqual(a.config_hash(), b.config_hash())

    def test_round_trips_through_dict(self):
        config = load_config("small_smoke_test")
        restored = CodeModelTrainingConfig.from_dict(config.to_dict())
        self.assertEqual(restored.config_hash(), config.config_hash())

    def test_load_config_accepts_a_direct_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "custom.yaml"
            path.write_text("name: custom\nbase_model:\n  model_id: some/model\n", encoding="utf-8")
            config = load_config(str(path))
            self.assertEqual(config.name, "custom")
            self.assertEqual(config.base_model.model_id, "some/model")


class HardwareDetectionTests(unittest.TestCase):
    def test_detect_hardware_never_raises_regardless_of_installed_packages(self):
        info = detect_hardware()
        self.assertIsInstance(info, HardwareInfo)
        self.assertIsInstance(info.cuda_available, bool)
        self.assertIsInstance(info.system_ram_total_mb, float)
        self.assertGreaterEqual(info.system_ram_total_mb, 0.0)

    def test_detect_hardware_reports_disk_free(self):
        info = detect_hardware(".")
        self.assertGreaterEqual(info.disk_free_mb, 0.0)

    def test_check_feasibility_never_raises_when_torch_or_network_unavailable(self):
        config = load_config("small_smoke_test")
        # An intentionally-unreachable model id forces the "could not
        # fetch/estimate base model size" path regardless of network state.
        config.base_model.model_id = "definitely/not-a-real-model-id-xyz"
        result = check_feasibility(config, min_free_disk_mb=0)
        self.assertIn(result.mode, ("LOCAL", "LOCAL_TRAINING_NOT_FEASIBLE"))
        self.assertIsInstance(result.reasons, list)

    def test_feasibility_result_includes_hardware_snapshot(self):
        config = load_config("small_smoke_test")
        config.base_model.model_id = "definitely/not-a-real-model-id-xyz"
        result = check_feasibility(config, min_free_disk_mb=0)
        self.assertIsInstance(result.hardware, HardwareInfo)


if __name__ == "__main__":
    unittest.main()
