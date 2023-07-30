"""Test config loading."""
# pylint: disable=unspecified-encoding

from textwrap import dedent

import pytest

from experiment_helpers import config as ehconfig


class _BaseConfig(ehconfig.BaseConfig):
    attribute = 21
    unchanged_attribute = 123

    @property
    def dynamic_attribute(self):
        """Dynamic attributes are a main benefit of a config object."""
        return f"{self.attribute}"

    def config_helper(self, arg):
        """The config can contain helper functions like formatters."""
        return f"{arg}"


class _UserConfig:
    attribute = 42


class _UserSubConfig(_BaseConfig):
    attribute = 42


class _UserConfigEmpty:
    ...


class _UserSubConfigEmpty(_BaseConfig):
    ...


def assert_config_content(config, expected_attribute):
    """Assert that the config has the expected content.

    Defaults need to be loaded first.
    """
    assert config.attribute == expected_attribute
    assert config.unchanged_attribute == 123
    assert config.dynamic_attribute == f"{expected_attribute}"
    assert config.config_helper(123) == "123"
    return True


@pytest.mark.parametrize("userconfig", [
    dict(attribute=42),    # as dict.
    _UserConfig,                # as class.
    _UserConfig(),              # as object.
    _UserSubConfig,             # Also works whether the config is already
    _UserSubConfig(),           # inheriting from the default config or not.
])
def test_with_updates(userconfig):
    """Test all ways the user can provide config updates."""
    # Test explicit loading.
    config = _BaseConfig.with_updates(userconfig)
    assert assert_config_content(config, 42)


@pytest.mark.parametrize("userconfig", [
    {},
    None,
    _UserConfigEmpty,
    _UserConfigEmpty(),
    _UserSubConfigEmpty,
    _UserSubConfigEmpty(),
])
def test_load_nothing(userconfig):
    """The functions also accepts no config."""
    # Test explicit loading.
    config = _BaseConfig.with_updates(userconfig)
    assert assert_config_content(config, 21)


def test_load_config_from_pyfile(tmp_path):
    """Test utilities to load config from file."""
    conf_path = tmp_path / "config.py"
    with open(conf_path, 'w') as file:
        file.write(dedent("""
        class Config:
            attribute = 123
        """))

    userconfig = ehconfig.load_config_from_pyfile(conf_path)
    config = _BaseConfig.with_updates(userconfig)
    assert_config_content(config, 123)
