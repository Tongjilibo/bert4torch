class _LazyAutoMapping(OrderedDict[type[PretrainedConfig], _LazyAutoMappingValue]):
    """
    " A mapping config to object (model or tokenizer for instance) that will load keys and values when it is accessed.

    Args:
        - config_mapping: The map model type to config class
        - model_mapping: The map model type to model (or tokenizer) class
    """

    def __init__(self, config_mapping, model_mapping) -> None:
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._model_mapping._model_mapping = self
        self._extra_content = {}
        self._modules = {}

    def __len__(self) -> int:
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key: type[PretrainedConfig]) -> _LazyAutoMappingValue:
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type, attr):
        module_name = model_type_to_module_name(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self) -> list[type[PretrainedConfig]]:
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key: type[PretrainedConfig], default: _T) -> Union[_LazyAutoMappingValue, _T]:
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __bool__(self) -> bool:
        return bool(self.keys())

    def values(self) -> list[_LazyAutoMappingValue]:
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping
        ]
        return mapping_values + list(self._extra_content.values())

    def items(self) -> list[tuple[type[PretrainedConfig], _LazyAutoMappingValue]]:
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping
            if key in self._config_mapping
        ]
        return mapping_items + list(self._extra_content.items())

    def __iter__(self) -> Iterator[type[PretrainedConfig]]:
        return iter(self.keys())

    def __contains__(self, item: type) -> bool:
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key: type[PretrainedConfig], value: _LazyAutoMappingValue, exist_ok=False) -> None:
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping and not exist_ok:
                raise ValueError(f"'{key}' is already used by a Transformers model.")

        self._extra_content[key] = value