# -*- coding: utf-8 -*-
# @Author  : LUYADONG977
from typing import Text, Dict, Any, Union, List

from ruamel import yaml
# import ruamel_yaml as yaml


class Configuration(object):

    configurations = {}

    @staticmethod
    def fix_yaml_loader() -> None:
        """Ensure that any string read by yaml is represented as unicode."""

        def construct_yaml_str(self, node):
            # Override the default string handling function
            # to always return unicode objects
            return self.construct_scalar(node)

        yaml.Loader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)
        yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", construct_yaml_str)

    @staticmethod
    def replace_environment_variables():
        """Enable yaml loader to process the environment variables in the yaml."""
        import re
        import os

        # eg. ${USER_NAME}, ${PASSWORD}
        env_var_pattern = re.compile(r"^(.*)\$\{(.*)\}(.*)$")
        yaml.add_implicit_resolver("!env_var", env_var_pattern)

        def env_var_constructor(loader, node):
            """Process environment variables found in the YAML."""
            value = loader.construct_scalar(node)
            expanded_vars = os.path.expandvars(value)
            if "$" in expanded_vars:
                not_expanded = [w for w in expanded_vars.split() if "$" in w]
                raise ValueError(
                    "Error when trying to expand the environment variables"
                    " in '{}'. Please make sure to also set these environment"
                    " variables: '{}'.".format(value, not_expanded)
                )
            return expanded_vars

        yaml.SafeConstructor.add_constructor("!env_var", env_var_constructor)

    @staticmethod
    def read_yaml(content: Text) -> Union[List[Any], Dict[Text, Any]]:
        """Parses yaml from a text.

         Args:
            content: A text containing yaml content.
        """
        Configuration.fix_yaml_loader()

        Configuration.replace_environment_variables()

        yaml_parser = yaml.YAML(typ="safe")
        yaml_parser.version = "1.2"
        yaml_parser.unicode_supplementary = True

        # noinspection PyUnresolvedReferences
        try:
            return yaml_parser.load(content) or {}
        except yaml.scanner.ScannerError:
            # A `ruamel.yaml.scanner.ScannerError` might happen due to escaped
            # unicode sequences that form surrogate pairs. Try converting the input
            # to a parsable format based on
            # https://stackoverflow.com/a/52187065/3429596.
            content = (
                content.encode("utf-8")
                    .decode("raw_unicode_escape")
                    .encode("utf-16", "surrogatepass")
                    .decode("utf-16")
            )
            return yaml_parser.load(content) or {}

    @staticmethod
    def read_file(filename: Text, encoding: Text = "utf-8") -> Any:
        """Read text from a file."""

        try:
            with open(filename, encoding=encoding) as f:
                return f.read()
        except FileNotFoundError:
            raise ValueError("File '{}' does not exist.".format(filename))

    @staticmethod
    def read_config_file(filename: Text) -> Dict[Text, Any]:
        """Parses a yaml configuration file. Content needs to be a dictionary

         Args:
            filename: The path to the file which should be read.
        """
        content = Configuration.read_yaml(Configuration.read_file(filename, "utf-8"))

        if content is None:
            return {}
        elif isinstance(content, dict):
            return content
        else:
            raise ValueError(
                "Tried to load invalid config file '{}'. "
                "Expected a key value mapping but found {}"
                ".".format(filename, type(content))
            )

    @classmethod
    def get(cls, configname, def_val):
        if configname in cls.configurations:
            return cls.configurations[configname]
        else:
            return def_val


class ConfigurationUtils:
    configurations = {}

    def __init__(self):
        pass

    @classmethod
    def get_port(cls):
        return cls.configurations["port"]

    @classmethod
    def get_zk_path_midintent(cls):
        return cls.configurations["zk_path_mid"]

    @classmethod
    def get_zk_path_entity(cls):
        return cls.configurations["zk_path_entity"]

    @classmethod
    def get_config(cls, config_name, def_val):
        if config_name in cls.configurations:
            return cls.configurations[config_name]
        else:
            return def_val

    @classmethod
    def get_nas_path(cls):
        return cls.configurations["model_path"]


if __name__ == '__main__':
    import src.config.constants as constants
    conf = Configuration.read_config_file('/Users/lvqi034/PycharmProjects/rec_news/conf/configurations.yml')
    bi_conf = conf.get(constants.CONFIG_BI_MODEL_KEY)
    model_path = bi_conf.get(constants.CONFIG_BI_MODEL_PATH_KEY)
    print(model_path)
    print(bi_conf.get("poolid")[0:5])

