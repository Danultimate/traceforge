"""TraceForge state serialiser.

Loud, actionable errors when state cannot be JSON-serialised. Users opt-out
explicitly via `traceforge.exclude(obj)` or implement
`__traceforge_serialise__(self) -> dict` on their class.
"""
from typing import Any


class TraceSerialiseError(Exception):
    def __init__(self, field: str, type_name: str, hint: str):
        super().__init__(
            f"\nTraceForge serialisation error:\n"
            f"  Field:  {field}\n"
            f"  Type:   {type_name} (not JSON-serialisable)\n"
            f"  Fix:    {hint}\n"
        )
        self.field = field
        self.type_name = type_name


def exclude(obj):
    """Mark an object as excluded from TraceForge state serialisation."""
    try:
        setattr(obj, "__traceforge_exclude__", True)
    except (AttributeError, TypeError):
        # Built-ins like int/str can't take new attributes — wrap them.
        return _ExcludedWrapper(obj)
    return obj


class _ExcludedWrapper:
    __traceforge_exclude__ = True

    def __init__(self, inner):
        self._inner = inner


def serialise_state(state: Any, slim: bool = False) -> dict | None:
    """Serialise agent state to a JSON-compatible dict.

    Raises TraceSerialiseError with actionable message on failure.
    """
    if state is None:
        return None
    if slim:
        return {"__slim__": True, "type": type(state).__name__}
    result = _safe_serialise(state, path="state")
    if not isinstance(result, dict):
        return {"value": result}
    return result


def _safe_serialise(obj: Any, path: str) -> Any:
    # Excluded objects
    if getattr(obj, "__traceforge_exclude__", False):
        inner_type = (
            type(obj._inner).__name__
            if isinstance(obj, _ExcludedWrapper)
            else type(obj).__name__
        )
        return {"__excluded__": True, "type": inner_type}

    # Custom serialiser
    if hasattr(obj, "__traceforge_serialise__"):
        return obj.__traceforge_serialise__()

    # Primitives
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # Dicts
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            try:
                result[str(k)] = _safe_serialise(v, f"{path}.{k}")
            except TraceSerialiseError:
                raise
            except Exception as e:
                raise TraceSerialiseError(
                    field=f"{path}.{k}",
                    type_name=type(v).__name__,
                    hint="Wrap with traceforge.exclude() or implement __traceforge_serialise__",
                ) from e
        return result

    # Lists / tuples
    if isinstance(obj, (list, tuple)):
        return [_safe_serialise(item, f"{path}[{i}]") for i, item in enumerate(obj)]

    # Pydantic models
    if hasattr(obj, "model_dump"):
        return _safe_serialise(obj.model_dump(), path)

    # Dataclasses
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return _safe_serialise(dataclasses.asdict(obj), path)

    raise TraceSerialiseError(
        field=path,
        type_name=type(obj).__name__,
        hint="Wrap with traceforge.exclude() or implement __traceforge_serialise__(self) -> dict",
    )
