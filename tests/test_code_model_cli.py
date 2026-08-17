"""Phase 25/29: CLI argument parsing for the one-command entry points.
Argument parsing itself needs no ML stack, so these run in any environment;
full functional execution of these CLIs was proven manually against
.venv-code-model (see the final report's smoke-test section)."""
import unittest

from training.code_model import evaluate as evaluate_cli
from training.code_model import export as export_cli
from training.code_model import start_learning as start_learning_cli
from training.code_model import train as train_cli
from training.code_model.benchmark import __main__ as benchmark_cli


class TrainCliTests(unittest.TestCase):
    def test_requires_config_and_dataset(self):
        parser = train_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parses_minimal_args(self):
        parser = train_cli.build_parser()
        args = parser.parse_args(["--config", "small_smoke_test", "--dataset", "v1.jsonl"])
        self.assertEqual(args.config, "small_smoke_test")
        self.assertEqual(args.dataset, "v1.jsonl")
        self.assertFalse(args.force)

    def test_force_flag(self):
        parser = train_cli.build_parser()
        args = parser.parse_args(["--config", "c", "--dataset", "d", "--force"])
        self.assertTrue(args.force)


class EvaluateCliTests(unittest.TestCase):
    def test_requires_model(self):
        parser = evaluate_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_default_config(self):
        parser = evaluate_cli.build_parser()
        args = parser.parse_args(["--model", "m1"])
        self.assertEqual(args.config, "qlora_7b")


class BenchmarkCliTests(unittest.TestCase):
    def test_requires_model(self):
        parser = benchmark_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parses_model_and_config(self):
        parser = benchmark_cli.build_parser()
        args = parser.parse_args(["--model", "m1", "--config", "small_smoke_test"])
        self.assertEqual(args.model, "m1")
        self.assertEqual(args.config, "small_smoke_test")


class ExportCliTests(unittest.TestCase):
    def test_requires_output(self):
        parser = export_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model", "m1"])

    def test_accepts_either_model_or_adapter(self):
        parser = export_cli.build_parser()
        args = parser.parse_args(["--adapter", "/path/to/adapter", "--output", "/path/to/out"])
        self.assertEqual(args.adapter, "/path/to/adapter")
        self.assertIsNone(args.model)

    def test_resolve_adapter_path_requires_model_or_adapter(self):
        with self.assertRaises(ValueError):
            export_cli._resolve_adapter_path(None, None)

    def test_resolve_adapter_path_prefers_explicit_adapter(self):
        self.assertEqual(export_cli._resolve_adapter_path(None, "/direct/path"), "/direct/path")


class StartLearningCliTests(unittest.TestCase):
    def test_requires_repository_root(self):
        parser = start_learning_cli.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_parses_repository_root(self):
        parser = start_learning_cli.build_parser()
        args = parser.parse_args(["--repository-root", "/some/repo"])
        self.assertEqual(args.repository_root, "/some/repo")
        self.assertIsNone(args.config)


if __name__ == "__main__":
    unittest.main()
