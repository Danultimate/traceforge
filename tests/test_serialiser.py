import dataclasses

import pytest
from pydantic import BaseModel

from traceforge import exclude, TraceSerialiseError
from traceforge.serialiser import serialise_state


def test_primitives_and_collections_roundtrip():
    state = {
        "n": 1,
        "f": 1.5,
        "s": "hi",
        "b": True,
        "none": None,
        "lst": [1, "two", [3]],
        "nested": {"a": {"b": [1, 2]}},
    }
    out = serialise_state(state)
    assert out == state


def test_pydantic_model_is_serialised():
    class M(BaseModel):
        a: int
        b: str

    out = serialise_state({"m": M(a=1, b="x")})
    assert out == {"m": {"a": 1, "b": "x"}}


def test_dataclass_is_serialised():
    @dataclasses.dataclass
    class D:
        a: int
        b: str

    out = serialise_state({"d": D(a=2, b="y")})
    assert out == {"d": {"a": 2, "b": "y"}}


def test_excluded_object_yields_placeholder():
    class C:
        pass

    c = C()
    out = serialise_state({"client": exclude(c)})
    assert out["client"] == {"__excluded__": True, "type": "C"}


def test_unserialisable_object_raises_with_field_path():
    class Opaque:
        pass

    with pytest.raises(TraceSerialiseError) as info:
        serialise_state({"top": {"nested": Opaque()}})
    msg = str(info.value)
    assert "state.top.nested" in msg
    assert "Opaque" in msg
    assert "traceforge.exclude" in msg


def test_custom_serialise_hook():
    class Custom:
        def __traceforge_serialise__(self):
            return {"shape": "round"}

    out = serialise_state({"obj": Custom()})
    assert out == {"obj": {"shape": "round"}}


def test_slim_mode_returns_metadata_only():
    out = serialise_state({"any": "thing"}, slim=True)
    assert out == {"__slim__": True, "type": "dict"}


def test_none_state_returns_none():
    assert serialise_state(None) is None
