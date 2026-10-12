#!/usr/bin/env python3
"""Fail if lib/bindings/python/Cargo.toml's [patch.crates-io] drifts from the root's.

The bindings crate declares its own [workspace] (pyo3 build constraint), so it is a
separate workspace root and does not inherit the root's patch table. maturin builds
that manifest, so a [patch.crates-io] present only in the root is silently ignored:
the wheel links the unpatched crates.io versions and still builds, installs, imports
and passes the version gate. The failure is invisible until runtime behaviour differs.
"""
import pathlib
import re
import sys

PATH_SPEC_RE = re.compile(r'\{ path = "([^"]+)" \}')
REPO = pathlib.Path.cwd().resolve()
PATCH_RE = re.compile(r"^\[patch\.crates-io\]\n(?:.*\n)*?(?=\n*\[|\Z)", re.M)


def section(path: pathlib.Path) -> dict[str, str] | None:
    m = PATCH_RE.search(path.read_text())
    if not m:
        return None
    entries = {}
    for line in m.group(0).splitlines()[1:]:
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, _, spec = line.partition("=")
        spec = " ".join(spec.split())
        # A `path =` source is relative to its own manifest, so the root and the
        # bindings manifest spell the same vendored crate differently. Compare
        # the directory it resolves to, not the string.
        pm = PATH_SPEC_RE.fullmatch(spec)
        if pm:
            spec = "{ path = %s }" % (path.parent / pm.group(1)).resolve().relative_to(REPO)
        entries[name.strip()] = spec
    return entries


def main() -> int:
    root = pathlib.Path("Cargo.toml")
    bindings = pathlib.Path("lib/bindings/python/Cargo.toml")
    want, got = section(root), section(bindings)

    if want is None:
        print("no [patch.crates-io] in the root manifest; nothing to mirror")
        return 0
    if got is None:
        print(f"ERROR: {root} has [patch.crates-io] but {bindings} has none.", file=sys.stderr)
        print("       maturin builds the bindings manifest, so the patch would be IGNORED", file=sys.stderr)
        print("       and the wheel would link unpatched crates.io versions.", file=sys.stderr)
        for k, v in want.items():
            print(f"       missing: {k} = {v}", file=sys.stderr)
        return 1
    if want != got:
        print(f"ERROR: [patch.crates-io] differs between {root} and {bindings}.", file=sys.stderr)
        for k in sorted(set(want) | set(got)):
            if want.get(k) != got.get(k):
                print(f"       {k}:\n         root     = {want.get(k, '<absent>')}\n"
                      f"         bindings = {got.get(k, '<absent>')}", file=sys.stderr)
        return 1

    print(f"[patch.crates-io] in sync across both manifests ({len(want)} entries):")
    for k, v in want.items():
        print(f"  {k} = {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
