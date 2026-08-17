import unittest

from brain.learning_models import DataQualityLabel
from brain.learning_validator import validate_variant, validate_variants
from brain.learning_variation import GeneratedVariant


def _valid_variant(variant_id="v1", seed=1) -> GeneratedVariant:
    return GeneratedVariant(
        variant_id=variant_id,
        description="off by one",
        manifest={"description": "off by one", "test_file": "tests/test_x.py"},
        before_files={
            "src.py": f"def f():\n    return {seed}\n",
            "tests/test_x.py": f"from src import f\ndef test_x():\n    assert f() == {seed + 1}\n",
        },
        after_files={
            "src.py": f"def f():\n    return {seed + 1}\n",
            "tests/test_x.py": f"from src import f\ndef test_x():\n    assert f() == {seed + 1}\n",
        },
    )


class ValidateVariantTests(unittest.TestCase):
    def test_genuine_before_fail_after_pass_is_verified(self):
        result = validate_variant(_valid_variant())
        self.assertTrue(result.verified)
        self.assertEqual(result.quality_label, DataQualityLabel.SYNTHETIC_VERIFIED.value)
        self.assertEqual(result.before_exit_code, 1)
        self.assertEqual(result.after_exit_code, 0)

    def test_missing_test_file_in_manifest_is_unverified(self):
        variant = _valid_variant()
        variant.manifest = {"description": "x"}
        result = validate_variant(variant)
        self.assertFalse(result.verified)
        self.assertIn("test_file", result.reason)

    def test_syntax_error_is_caught_before_running_pytest(self):
        variant = _valid_variant()
        variant.before_files["src.py"] = "def f(:\n    return 1\n"
        result = validate_variant(variant)
        self.assertFalse(result.verified)
        self.assertIn("syntax error", result.reason)
        self.assertEqual(result.quality_label, DataQualityLabel.SYNTHETIC_DERIVED.value)

    def test_before_that_already_passes_is_not_a_real_repro_and_is_rejected(self):
        variant = _valid_variant()
        # "fix" the before/ files too, so the test passes even before the fix
        variant.before_files["src.py"] = variant.after_files["src.py"]
        result = validate_variant(variant)
        self.assertFalse(result.verified)
        self.assertIn("did not cleanly fail", result.reason)

    def test_after_that_still_fails_is_rejected(self):
        variant = _valid_variant()
        variant.after_files["src.py"] = variant.before_files["src.py"]  # "fix" doesn't actually fix
        result = validate_variant(variant)
        self.assertFalse(result.verified)
        self.assertIn("did not pass", result.reason)

    def test_empty_file_set_is_rejected(self):
        variant = _valid_variant()
        variant.before_files = {}
        result = validate_variant(variant)
        self.assertFalse(result.verified)


class ValidateVariantsBatchTests(unittest.TestCase):
    def test_verified_and_unverified_are_kept_strictly_separate(self):
        good = _valid_variant("v-good", seed=1)
        bad = _valid_variant("v-bad", seed=2)
        bad.before_files["src.py"] = "def f(:\n"  # syntax error -> unverifiable
        verified, unverified = validate_variants([good, bad])
        self.assertEqual([r.variant_id for r in verified], ["v-good"])
        self.assertEqual([r.variant_id for r in unverified], ["v-bad"])
        for r in verified:
            self.assertEqual(r.quality_label, DataQualityLabel.SYNTHETIC_VERIFIED.value)
        for r in unverified:
            self.assertEqual(r.quality_label, DataQualityLabel.SYNTHETIC_DERIVED.value)


if __name__ == "__main__":
    unittest.main()
