"""Settings come from OWL_* variables, so a typo in owl.env must say what is wrong."""

import re
import unittest
from pathlib import Path
from unittest import mock

from owl.config import Config

REPO = Path(__file__).resolve().parent.parent


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            config = Config()
        self.assertEqual(config.retention_days, 30)
        self.assertEqual(config.unidentified_days, 3)
        self.assertEqual(config.low_disk_percent, 10)
        self.assertEqual(config.api_host, "127.0.0.1")
        self.assertEqual(config.cors_origins, ("*",))

    def test_values_are_read_from_the_environment(self):
        env = {"OWL_RETENTION_DAYS": " 14 ", "OWL_SPECIES_MIN_SCORE": "0.9", "OWL_HFLIP": "Yes",
               "OWL_NOTIFY_MUTE": "Person, Eastern Gray Squirrel"}
        with mock.patch.dict("os.environ", env, clear=True):
            config = Config()
        self.assertEqual((config.retention_days, config.species_min_score, config.hflip), (14, 0.9, True))
        self.assertEqual(config.notify_mute, ("person", "eastern gray squirrel"))

    def test_blank_values_mean_the_default(self):
        env = {"OWL_RETENTION_DAYS": "", "OWL_MOTION_MIN_AREA": "  ", "OWL_DETECT_PEOPLE": ""}
        with mock.patch.dict("os.environ", env, clear=True):
            config = Config()
        self.assertEqual((config.retention_days, config.motion_min_area, config.detect_people), (30, 0.0015, True))

    def test_a_bad_number_names_the_variable(self):
        for name, value in (("OWL_RETENTION_DAYS", "30 days"), ("OWL_RETENTION_DAYS", "2.5"),
                            ("OWL_SPECIES_MIN_SCORE", "0.95     # confidence needed")):
            with self.subTest(name=name, value=value), mock.patch.dict("os.environ", {name: value}, clear=True):
                with self.assertRaises(SystemExit) as raised:
                    Config()
                self.assertIn(name, str(raised.exception))


class ExampleFileTest(unittest.TestCase):
    """owl.env.example is what people copy from, so it must be accurate and safe to uncomment."""

    @classmethod
    def setUpClass(cls):
        cls.lines = (REPO / "owl.env.example").read_text().splitlines()

    def settings(self):
        """Every `NAME=value` line, commented out or not."""
        for line in self.lines:
            match = re.fullmatch(r"#?\s*(OWL_[A-Z_]+)=(.*)", line)
            if match:
                yield match[1], match[2]

    def test_no_value_carries_an_inline_comment(self):
        # systemd does not strip these, so uncommenting such a line would break the setting.
        for name, value in self.settings():
            self.assertNotIn("#", value, name)

    def test_every_setting_the_code_reads_is_documented(self):
        source = (REPO / "owl" / "config.py").read_text()
        read = set(re.findall(r'"(OWL_[A-Z_]+)"', source))
        documented = {name for name, _ in self.settings()}
        self.assertEqual(read - documented, set())
        self.assertEqual(documented - read, set(), "documented but never read")

    def test_documented_defaults_are_the_real_defaults(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            default = Config()
        for name, value in self.settings():
            if not value or name in ("OWL_API_SECRET", "OWL_NTFY_TOPIC", "OWL_SITE_URL"):
                continue
            with self.subTest(name=name), mock.patch.dict("os.environ", {name: value}, clear=True):
                self.assertEqual(Config(), default)

    def test_the_installer_can_fill_in_the_secret_and_the_topic(self):
        text = "\n".join(self.lines)
        self.assertRegex(text, r"(?m)^OWL_API_SECRET=$")
        self.assertRegex(text, r"(?m)^OWL_NTFY_TOPIC=$")


if __name__ == "__main__":
    unittest.main()
