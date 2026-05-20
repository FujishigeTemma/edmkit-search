"""Generate API reference Markdown files from docstrings using griffe2md."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

from griffe import Alias, Attribute, ExprList, GriffeLoader, Module, Object, Parser
from griffe import TypeAlias as GriffeTypeAlias
from griffe2md import ConfigDict, default_config, render_object_docs

OUTPUT_DIR = Path(__file__).parent / "src" / "content" / "docs" / "reference"


class ModuleMeta(NamedTuple):
    description: str
    order: int


# Subpackages under edmkit.search. The order here matches the conceptual
# progression: data in → state primitives → expansion → scoring → strategy.
MODULES: dict[str, ModuleMeta] = {
    "dataset": ModuleMeta("Time-series dataset containers and transforms.", 1),
    "state": ModuleMeta("Search-state primitives.", 2),
    "neighborhood": ModuleMeta("Expanders that map a batch of parent states to children.", 3),
    "energy": ModuleMeta("Scoring functions for batches of states, with deferred-job plans.", 4),
    "strategy": ModuleMeta("Per-step transitions and the top-level search runner.", 5),
}

CONFIG: ConfigDict = {
    **default_config,
    "docstring_style": "numpy",
    "heading_level": 2,
    "show_root_heading": True,
    "show_root_full_path": False,
    "show_root_members_full_path": False,
    "show_object_full_path": False,
    "show_if_no_docstring": False,
    "show_signature_annotations": True,
    "separate_signature": True,
    "members_order": "source",
    "docstring_section_style": "table",
}


def resolve_member(module: Object, name: str) -> Object:
    """Resolve a member by name, following aliases and looking into submodules."""
    member = module.members.get(name)
    if member is None:
        raise KeyError(f"{name} not found in {module.path}")
    if isinstance(member, Alias):
        return member.final_target
    # When a submodule shadows the imported function (same name),
    # look for the function inside the submodule.
    if isinstance(member, Module) and name in member.members:
        return resolve_member(member, name)
    return member


def export_names(module: Object) -> list[str] | None:
    """Return the names listed in ``__all__``, or None if not declared."""
    all_attr = module.members.get("__all__")
    if all_attr is None:
        return None
    if not isinstance(all_attr, Attribute):
        raise TypeError(f"Expected __all__ to be Attribute, got {type(all_attr).__name__}")
    if not isinstance(all_attr.value, ExprList):
        raise TypeError(f"Expected __all__ value to be ExprList, got {type(all_attr.value).__name__}")
    return [str(element).strip("'\"") for element in all_attr.value.elements]


def docstring_summary(obj: Object) -> str:
    """First line of the parsed docstring, or empty string when absent."""
    if obj.docstring is None:
        return ""
    return obj.docstring.parsed[0].value.split("\n")[0]


def render_overview_table(items: list[tuple[str, Object]]) -> str:
    """Render a Markdown table listing exported functions and classes."""
    rows = [f"[`{name}`](#{name}) | {docstring_summary(obj)}" for name, obj in items]
    return "**Functions:**\n\nName | Description\n---- | -----------\n" + "\n".join(rows)


def render_type_aliases_table(items: list[tuple[str, GriffeTypeAlias]]) -> str:
    """Render the Type Aliases overview table and per-alias detail blocks."""
    rows = [f"[`{name}`](#{alias.path}) | {docstring_summary(alias)}" for name, alias in items]
    table = "**Type Aliases:**\n\nName | Description\n---- | -----------\n" + "\n".join(rows)

    details = []
    for name, alias in items:
        body = alias.docstring.value if alias.docstring else ""
        details.append(f"### `{name}` {{#{alias.path}}}\n\n```python\ntype {name} = {alias.value}\n```\n\n{body}")
    return "\n\n".join([table, *details])


def render_package_exports(module: Object) -> str:
    """Render ``__all__`` exports of a package, bucketing type aliases separately.

    griffe2md has no template for type aliases, so they are pulled out of the
    main ``render_object_docs`` path and rendered by hand. Functions and
    classes go through griffe2md as usual.
    """
    names = export_names(module)
    if names is None:
        return render_object_docs(module, CONFIG)

    show_undocumented = bool(CONFIG.get("show_if_no_docstring", False))

    type_aliases: list[tuple[str, GriffeTypeAlias]] = []
    other: list[tuple[str, Object]] = []
    for name in names:
        obj = resolve_member(module, name)
        if isinstance(obj, GriffeTypeAlias):
            if show_undocumented or obj.docstring is not None:
                type_aliases.append((name, obj))
        else:
            other.append((name, obj))

    sections: list[str] = []
    if type_aliases:
        sections.append(render_type_aliases_table(type_aliases))
    if other:
        sections.append(render_overview_table(other))
        sections.extend(render_object_docs(obj, CONFIG) for _, obj in other)
    return "\n\n".join(sections)


def render_module(module: Object) -> str:
    """Render a module via ``__all__`` when present, else fall back to griffe2md."""
    if isinstance(module, Module) and export_names(module) is not None:
        return render_package_exports(module)
    return render_object_docs(module, CONFIG)


def write_reference(module_name: str, meta: ModuleMeta, body: str) -> Path:
    """Write a single reference file with Starlight frontmatter."""
    frontmatter = f"""---
title: {module_name}
description: {meta.description}
sidebar:
  order: {meta.order}
---

"""
    path = OUTPUT_DIR / (module_name.replace("_", "-") + ".md")
    path.write_text(frontmatter + body)
    return path


def main() -> None:
    loader = GriffeLoader(
        search_paths=CONFIG.get("search_paths", []) + sys.path,
        docstring_parser=Parser("numpy"),
        docstring_options=CONFIG.get("docstring_options", {}),
    )
    package = loader.load("edmkit.search")
    loader.resolve_aliases(external=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for module_name, meta in MODULES.items():
        module = package.modules[module_name]
        path = write_reference(module_name, meta, render_module(module))
        print(f"  {path.name}")

    print(f"Generated {len(MODULES)} files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
