from enum import Enum

import yaml


class YamlEnum(str, Enum):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for dumper in (yaml.SafeDumper, yaml.Dumper):
            dumper.add_multi_representer(
                cls, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v.value)
            )
