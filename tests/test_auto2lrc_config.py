import os
import unittest
from unittest import mock

from auto2lrc import Auto2Lrc


class Auto2LrcConfigTests(unittest.TestCase):
    def test_default_model_is_turbo(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Auto2Lrc().model_name, "turbo")

    def test_model_can_be_overridden_by_environment(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"WHISPER_MODEL_NAME": "large-v3"},
            clear=True,
        ):
            self.assertEqual(Auto2Lrc().model_name, "large-v3")

    def test_explicit_model_takes_precedence(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"WHISPER_MODEL_NAME": "large-v3"},
            clear=True,
        ):
            self.assertEqual(Auto2Lrc(model_name="small").model_name, "small")


if __name__ == "__main__":
    unittest.main()
