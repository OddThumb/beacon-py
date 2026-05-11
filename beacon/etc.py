import numpy as np
from typing import Dict, Any
from pympler import asizeof

# Inspect function content
def inspect_func(func):
    """
    Print the source code of a given function.

    Args:
        func (callable): Function object to inspect.

    Returns:
        None
    """
    import inspect

    print(inspect.getsource(func))

def format_size(size_bytes):
    """Convert bytes to the most readable unit (MB, KB, B)"""
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.4f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes} B"


def inspect_size(obj):
    total_size = asizeof.asizeof(obj)
    print(f"--- Object Size Inspection (Total: {format_size(total_size)}) ---")

    # 1. For dictionaries: Show keys and their respective value sizes
    if isinstance(obj, dict):
        for key, value in obj.items():
            size = asizeof.asizeof(value)
            print(f"[Key] {str(key):<20} | {format_size(size)}")

    # 2. For other iterables (excluding strings/bytes): Show index-based sizes
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        for i, item in enumerate(obj):
            size = asizeof.asizeof(item)
            print(f"[Index] {i:<18} | {format_size(size)}")

    # 3. For non-iterable single objects
    else:
        obj_type = type(obj).__name__
        print(f"[Single Object] {obj_type:<11} | {format_size(asizeof.asizeof(obj))}")


class _SummaryStats(dict):
    """Dict-like stats holder: formatted in __str__, silent in __repr__ to avoid double-echo."""

    def _formatted(self) -> str:
        # Compute label width
        label_width = max(len(k) for k in self.keys())

        # Format values
        def _fmt(k: str, v: Any) -> str:
            return f"{int(v):d}" if k == "NA's" else f"{float(v):+.3e}"

        # Compute value width for alignment
        value_width = max(len(_fmt(k, v)) for k, v in self.items())
        # Build lines
        lines = [
            f"{k:>{label_width}} {_fmt(k, self[k]):>{value_width}}" for k in self.keys()
        ]
        lines.append(f"[dict object]")

        return "\n".join(lines)

    def __str__(self) -> str:
        # Pretty (human) display
        return self._formatted()

    def __repr__(self) -> str:
        # Keep notebook/REPL from echoing again when unassigned
        return ""


def summary(x) -> _SummaryStats:
    """
    R-style summary with aligned printing.
    Always prints once; returns a dict-like object for programmatic use.
    """
    # Ensure 1D float64
    arr = np.asarray(x, dtype=np.float64).ravel()
    nan_count = int(np.isnan(arr).sum())
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        raise ValueError("All values are NaN.")

    stats: Dict[str, Any] = {
        "Min.": float(np.min(valid)),
        "1st Qu.": float(np.percentile(valid, 25)),
        "Median": float(np.median(valid)),
        "Mean": float(np.mean(valid)),
        "Std.": float(np.std(valid, ddof=1)),
        "3rd Qu.": float(np.percentile(valid, 75)),
        "Max.": float(np.max(valid)),
        "NA's": nan_count,
    }

    out = _SummaryStats(stats)
    print(out)  # Triggers __str__; prints exactly once whether assigned or not
    return out


class Rist:
    """
    Flexible R style list-like container supporting optional named elements.

    Features:
        - Access elements by index or name.
        - Supports nested Rist objects.
        - Pretty string representation with tree structure.
        - Attribute access for named elements.
        - Safe combination of multiple Rist objects without duplicate names.

    Attributes:
        values (list): List of stored elements.
        names (list): List of names or None for unnamed elements.
        _name_to_index (dict): Mapping of names to indices for fast lookup.
    """
    __slots__ = ["values", "names", "_name_to_index"]

    def __init__(self, *args, **kwargs):
        """
        Initialize a Rist object.

        Args:
            *args: Positional arguments (unnamed elements or a single dict).
            **kwargs: Keyword arguments (named elements).
        """
        # Special case: one dict argument with no kwargs
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            dict_items = args[0]
            self.values = list(dict_items.values())
            self.names = list(dict_items.keys())
        else:
            self.values = list(args) + list(kwargs.values())
            self.names = [None] * len(args) + list(kwargs.keys())

        self._name_to_index = {
            name: idx for idx, name in enumerate(self.names) if name is not None
        }

    def _rebuild_index(self):
        """
        Rebuild internal name-to-index mapping after deletion or reordering.
        """
        self._name_to_index = {
            name: idx for idx, name in enumerate(self.names) if name is not None
        }

    def __getitem__(self, key):
        """
        Retrieve an element by index or name.

        Args:
            key (int or str): Index or name of the element.

        Returns:
            Any: The requested element.

        Raises:
            KeyError: If the name does not exist.
            TypeError: If the key is not int or str.
        """
        if isinstance(key, int):
            return self.values[key]
        elif isinstance(key, str):
            idx = self._name_to_index.get(key)
            if idx is None:
                raise KeyError(f"Name '{key}' not found")
            return self.values[idx]
        else:
            raise TypeError("Key must be int or str")

    def __setitem__(self, key, value):
        """
        Set an element by index or name.

        Args:
            key (int or str): Index or name of the element.
            value (Any): Value to assign.

        Behavior:
            - If key is int:
                * Sets the value at the given index.
                * Automatically extends the list with None if index is beyond current length.
            - If key is str:
                * Overwrites existing value if name exists.
                * Appends as a new named element if name does not exist.

        Raises:
            IndexError: If key is a negative integer.
            TypeError: If key is neither int nor str.
        """
        if isinstance(key, int):
            if key < 0:
                raise IndexError("Negative indexing not supported")
            while key >= len(self.values):
                self.values.append(None)
                self.names.append(None)
            self.values[key] = value
        elif isinstance(key, str):
            idx = self._name_to_index.get(key)
            if idx is not None:
                self.values[idx] = value
            else:
                self.values.append(value)
                self.names.append(key)
                self._name_to_index[key] = len(self.values) - 1
        else:
            raise TypeError("Key must be int or str")

    def __delitem__(self, key):
        """
        Delete item by name from a named Rist.

        Args:
            key (str): Name of the item to delete.

        Raises:
            KeyError: If the key does not exist.
        """
        if key not in self.names:
            raise KeyError(f"Key '{key}' not found in Rist.")

        idx = self._name_to_index[key]

        # Remove entry
        del self.values[idx]
        del self.names[idx]
        self._rebuild_index()

    def __getattr__(self, name):
        """
        Enable attribute-like access to named elements.

        Args:
            name (str): Name of the element.

        Returns:
            Any: The requested element.

        Raises:
            AttributeError: If the name does not exist.
        """
        idx = self._name_to_index.get(name)
        if idx is not None:
            return self.values[idx]
        raise AttributeError(f"'Rist' object has no attribute '{name}'")

    def __contains__(self, key):
        """Check if a named element exists."""
        if isinstance(key, str):
            return key in self._name_to_index
        return key in self.values

    def __iter__(self):
        return iter(self.values)

    def __dir__(self):
        base = super().__dir__()
        return list(base) + [n for n in self.names if n is not None]

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return f"Rist({self.names})"

    def __str__(self):
        """
        Return a tree-style string representation of contents.

        Returns:
            str: Formatted string showing the nested structure.
        """
        return self._str_with_indent(name="(root)", parent_prefix="", is_last=True)

    def _str_with_indent(self, name, parent_prefix, is_last):
        """
        Helper to recursively build the tree-style string representation.

        Args:
            name (str): Name or label of the current node.
            parent_prefix (str): Prefix for alignment.
            is_last (bool): Whether this is the last child.

        Returns:
            str: Multi-line string for this node and children.
        """
        lines = []
        connector = "└── " if is_last else "├── "

        # Determine label text
        label_text = f"{connector}.{name} (Rist)" if name != "(root)" else f".(Rist)"
        lines.append(parent_prefix + label_text)

        # Build new prefix for children
        new_prefix = parent_prefix + ("    " if is_last else "│   ")

        for i, (child_name, child_value) in enumerate(zip(self.names, self.values)):
            is_child_last = i == len(self.values) - 1
            if child_name is None:
                child_label = f"[{i}]"
            else:
                child_label = child_name

            if isinstance(child_value, Rist):
                # Recursively format nested Rist
                lines.append(
                    child_value._str_with_indent(
                        name=child_label,
                        parent_prefix=new_prefix,
                        is_last=is_child_last,
                    )
                )
            else:
                # Leaf node
                connector_child = "└── " if is_child_last else "├── "
                line = (
                    new_prefix
                    + f"{connector_child}.{child_label} ({type(child_value).__name__})"
                )
                lines.append(line)
                # Indent value lines
                repr_lines = repr(child_value).splitlines()
                for rline in repr_lines:
                    lines.append(
                        new_prefix + ("    " if is_child_last else "│   ") + rline
                    )
        return "\n".join(lines)

    __repr__ = __str__

    def __add__(self, other):
        """
        Combine two Rist objects into a new Rist.

        Args:
            other (Rist): Another Rist to combine.

        Returns:
            Rist: New combined Rist object.

        Raises:
            ValueError: If duplicate names are detected.
        """
        if not isinstance(other, Rist):
            return NotImplemented

        # Check duplicate names
        self_names_set = {n for n in self.names if n is not None}
        other_names_set = {n for n in other.names if n is not None}
        duplicates = self_names_set & other_names_set

        if duplicates:
            dup_list = ", ".join(f"'{d}'" for d in duplicates)
            raise ValueError(f"Duplicate name(s) {dup_list} detected during addition.")

        # Combine
        combined = Rist()
        combined.values = self.values + other.values
        combined.names = self.names + other.names
        combined._name_to_index = {
            n: i for i, n in enumerate(combined.names) if n is not None
        }
        return combined

    def append(self, *args, **kwargs):
        """
        Append an element to the Rist.

        Usage:
            - append(obj): Append unnamed element.
            - append(name=obj): Append named element (must be unique).

        Args:
            *args: Single unnamed element.
            **kwargs: Single named element.

        Raises:
            ValueError: If both positional and keyword arguments are provided,
                        more than one argument is provided,
                        or name duplicates exist.
        """
        if args and kwargs:
            raise ValueError("Cannot use both positional and keyword arguments.")
        if len(args) > 1:
            raise ValueError("Only one positional argument allowed.")
        if len(kwargs) > 1:
            raise ValueError("Only one keyword argument allowed.")

        if args:
            obj = args[0]
            self.values.append(obj)
            self.names.append(None)
        elif kwargs:
            name, obj = next(iter(kwargs.items()))
            if name in self._name_to_index:
                raise ValueError(f"Name '{name}' already exists in this Rist.")
            self.values.append(obj)
            self.names.append(name)
            self._name_to_index[name] = len(self.values) - 1
        else:
            raise ValueError("No object provided to append.")

    def to_dict(self, flat: bool = False) -> dict:
        """
        Convert Rist object to dictionary.

        Args:
            flat (bool): If True, flatten one-level nested Rist structure. Defaults to False.

        Returns:
            dict: Dictionary representation.
        """
        result = {}
        for name, val in zip(self.names, self.values):
            if name is None:
                continue  # Skip unnamed elements
            if flat and isinstance(val, Rist):
                # Merge flattened sub-Rist into top-level dict
                subdict = val.to_dict(flat=True)
                result.update(subdict)
            else:
                result[name] = val
        return result

    def copy(self) -> "Rist":
        has_names = any(name is not None for name in self.names)

        if has_names:
            named = {
                name: val.copy() if isinstance(val, Rist) else val
                for name, val in zip(self.names, self.values)
                if name is not None
            }
            return Rist(**named)
        else:
            unnamed = [
                val.copy() if isinstance(val, Rist) else val for val in self.values
            ]
            return Rist(*unnamed)
        
    def __getstate__(self):
        """
        For serialize in pickle
        """
        return {
            "values": self.values,
            "names": self.names,
            "_name_to_index": self._name_to_index,
        }
    
    def __setstate__(self, state):
        """
        For recovery from serialize in pickle
        """
        self.values = state["values"]
        self.names = state["names"]
        self._name_to_index = state["_name_to_index"]
    
    def save(self, path: str) -> None:
        """Save Rist object to file using pickle."""
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "Rist":
        """Load Rist object from file."""
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)

