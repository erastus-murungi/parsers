# Copyright 2023 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen Dictionary."""

from typing import Any, Dict, KeysView, Mapping, Tuple, TypeVar, Union, ValuesView


class FrozenKeysView(KeysView):
    """A wrapper for a more useful repr of the keys in a frozen dict."""

    def __repr__(self):
        return f"frozen_dict_keys({list(self)})"


class FrozenValuesView(ValuesView):
    """A wrapper for a more useful repr of the values in a frozen dict."""

    def __repr__(self):
        return f"frozen_dict_values({list(self)})"


K = TypeVar("K")
V = TypeVar("V")


def _indent(x, num_spaces):
    indent_str = " " * num_spaces
    lines = x.split("\n")
    assert not lines[-1]
    # skip the final line because it's empty and should not be indented.
    return "\n".join(indent_str + line for line in lines[:-1]) + "\n"


class FrozenDict(Mapping[K, V]):
    """An immutable variant of the Python dict."""

    __slots__ = ("_dict", "_hash")

    def __init__(
        self, *args, __unsafe_skip_copy__=False, **kwargs
    ):  # pylint: disable=invalid-name
        # make sure the dict is as
        xs = dict(*args, **kwargs)
        if __unsafe_skip_copy__:
            self._dict = xs
        else:
            self._dict = _prepare_freeze(xs)

        self._hash = None

    def __getitem__(self, key):
        v = self._dict[key]
        if isinstance(v, dict):
            return FrozenDict(v)
        return v

    def __setitem__(self, key, value):
        raise ValueError("FrozenDict is immutable.")

    def __contains__(self, key):
        return key in self._dict

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return self.pretty_repr()

    def __reduce__(self):
        return FrozenDict, (self.unfreeze(),)

    def pretty_repr(self, num_spaces=4):
        """Returns an indented representation of the nested dictionary."""

        def pretty_dict(x):
            if not isinstance(x, dict):
                return repr(x)
            rep = ""
            for key, val in x.items():
                rep += f"{key}: {pretty_dict(val)},\n"
            if rep:
                return "{\n" + _indent(rep, num_spaces) + "}"
            else:
                return "{}"

        return f"FrozenDict({pretty_dict(self._dict)})"

    def __hash__(self):
        if self._hash is None:
            h = 0
            for key, value in self.items():
                h ^= hash((key, value))
            self._hash = h
        return self._hash

    def copy(self, add_or_replace: Mapping[K, V]) -> "FrozenDict[K, V]":
        """Create a new FrozenDict with additional or replaced entries."""
        return type(self)({**self, **unfreeze(add_or_replace)})  # type: ignore[arg-type]

    def keys(self):
        return FrozenKeysView(self)

    def values(self):
        return FrozenValuesView(self)

    def items(self):
        for key in self._dict:
            yield (key, self[key])

    def pop(self, key: K) -> Tuple["FrozenDict[K, V]", V]:
        """Create a new FrozenDict where one entry is removed.

        Example::

          state, params = variables.pop('params')

        Args:
          key: the key to remove from the dict
        Returns:
          A pair with the new FrozenDict and the removed value.
        """
        value = self[key]
        new_dict = dict(self._dict)
        new_dict.pop(key)
        new_self = type(self)(new_dict)
        return new_self, value

    def unfreeze(self) -> Dict[K, V]:
        """Unfreeze this FrozenDict.

        Returns:
          An unfrozen version of this FrozenDict instance.
        """
        return unfreeze(self)

    @classmethod
    def tree_unflatten(cls, keys, values):
        # data is already deep copied due to tree map mechanism
        # we can skip the deep copy in the constructor
        return cls({k: v for k, v in zip(keys, values)}, __unsafe_skip_copy__=True)


def _prepare_freeze(xs: Any) -> Any:
    """Deep copy unfrozen dicts to make the dictionary FrozenDict safe."""
    if isinstance(xs, FrozenDict):
        # we can safely ref share the internal state of a FrozenDict
        # because it is immutable.
        return xs._dict  # pylint: disable=protected-access
    if not isinstance(xs, dict):
        # return a leaf as is.
        return xs
    # recursively copy dictionary to avoid ref sharing
    return {key: _prepare_freeze(val) for key, val in xs.items()}


def freeze(xs: Mapping[Any, Any]) -> FrozenDict[Any, Any]:
    """Freeze a nested dict.

    Makes a nested `dict` immutable by transforming it into `FrozenDict`.

    Args:
      xs: Dictionary to freeze (a regualr Python dict).
    Returns:
      The frozen dictionary.
    """
    return FrozenDict(xs)


def unfreeze(x: Union[FrozenDict, Dict[str, Any]]) -> Dict[Any, Any]:
    """Unfreeze a FrozenDict.

    Makes a mutable copy of a `FrozenDict` mutable by transforming
    it into (nested) dict.

    Args:
      x: Frozen dictionary to unfreeze.
    Returns:
      The unfrozen dictionary (a regular Python dict).
    """
    if isinstance(x, FrozenDict):
        # deep copy internal state of a FrozenDict
        # the dict branch would also work here but
        # it is much less performant because jax.tree_util.tree_map
        # uses an optimized C implementation.
        return jax.tree_util.tree_map(lambda y: y, x._dict)  # type: ignore
    elif isinstance(x, dict):
        ys = {}
        for key, value in x.items():
            ys[key] = unfreeze(value)
        return ys
    else:
        return x


def pretty_repr(x: Any, num_spaces: int = 4) -> str:
    """Returns an indented representation of the nested dictionary.
    This is a utility function that can act on either a FrozenDict or
    regular dict and mimics the behavior of `FrozenDict.pretty_repr`.
    If x is any other dtype, this function will return `repr(x)`.

    Args:
      x: the dictionary to be represented
      num_spaces: the number of space characters in each indentation level
    Returns:
      An indented string representation of the nested dictionary.
    """

    if isinstance(x, FrozenDict):
        return x.pretty_repr()
    else:

        def pretty_dict(x):
            if not isinstance(x, dict):
                return repr(x)
            rep = ""
            for key, val in x.items():
                rep += f"{key}: {pretty_dict(val)},\n"
            if rep:
                return "{\n" + _indent(rep, num_spaces) + "}"
            else:
                return "{}"

        return pretty_dict(x)
