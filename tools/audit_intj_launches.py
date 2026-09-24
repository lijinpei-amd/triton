"""Check that every remaining indexed Python JIT launch is documented.

Run from anywhere with ``python tools/audit_intj_launches.py``. The two small
allowlists are ordinary indexed calls and examples inside docstrings.
"""

import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "intj_launch_exceptions.json"
EMBEDDED_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\[[^\n]*?\]\s*\(")

# Indexed Python callbacks and a JIT-body function call, not host launches.
NON_LAUNCH_CALLS = Counter({
    ("python/test/unit/language/test_frontend.py", "fns"): 1,
    ("python/test/unit/language/test_tensor_descriptor.py", "REDUCE_OP"): 1,
    ("python/triton/_test_runner.py", "{'unit': _unit, 'gsan': _gsan, 'gluon': _gluon}"): 1,
    ("python/triton/compiler/compiler.py", "actives"): 1,
    ("python/triton/language/semantic.py", "self.builder.codegen_fns"): 2,
    ("python/triton/runtime/driver.py", "active_drivers"): 1,
    ("python/triton_kernels/tests/test_distributed.py", "factories"): 1,
    ("third_party/amd/python/test/cdna5_consan_helper.py", "tests"): 1,
})

# Launch-shaped prose in multiline strings, not executable embedded scripts.
NON_LAUNCH_EXAMPLES = Counter({
    ("python/triton/language/core.py", "inplace_abs"): 1,
    ("python/triton/runtime/jit.py", "fn"): 1,
    ("python/triton_kernels/triton_kernels/testing.py", "A"): 1,
    ("third_party/amd/python/test/test_gluon_cdna5_consan.py", "smem"): 1,
    ("third_party/amd/python/test/test_mxfp_gemm_tdm.py", "shape"): 1,
    ("third_party/proton/proton/scope.py", "foo"): 1,
    ("third_party/proton/proton/state.py", "foo"): 1,
})


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["version"] == 1
    documented = Counter((e["path"], e["line"], e["target"]) for e in manifest["exceptions"])
    assert len(documented) == len(manifest["exceptions"]), "duplicate exception entry"
    assert all(e["reason"] and e["detail"] for e in manifest["exceptions"])

    files = subprocess.check_output(["git", "ls-files", "--", "*.py"], cwd=ROOT, text=True).splitlines()
    launches = Counter()
    embedded = Counter()
    ignored_calls = Counter()
    ignored_examples = Counter()
    for path in files:
        source = (ROOT / path).read_text()
        lines = source.splitlines()
        tree = ast.parse(source, filename=path)
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
                target = ast.unparse(node.func.value)
                key = (path, target)
                if key in NON_LAUNCH_CALLS:
                    ignored_calls[key] += 1
                else:
                    launches[(path, node.lineno, target)] += 1
            if isinstance(node, (ast.Constant, ast.JoinedStr)) and not isinstance(parents.get(node), ast.JoinedStr):
                if isinstance(node, ast.Constant) and not isinstance(node.value, str):
                    continue
                if not hasattr(node, "lineno"):
                    continue
                for line in range(node.lineno + 1, node.end_lineno):
                    for match in EMBEDDED_CALL.finditer(lines[line - 1]):
                        key = (path, match.group(1))
                        if key in NON_LAUNCH_EXAMPLES:
                            ignored_examples[key] += 1
                        else:
                            embedded[(path, line, match.group(1))] += 1

    found = launches + embedded
    problems = []
    for label, actual, expected in (
        ("ordinary call allowlist", ignored_calls, NON_LAUNCH_CALLS),
        ("prose allowlist", ignored_examples, NON_LAUNCH_EXAMPLES),
    ):
        if actual != expected:
            problems.append(f"{label} drift: {actual - expected} added, {expected - actual} missing")
    for label, difference in (("unlisted launch", found - documented), ("stale exception", documented - found)):
        problems.extend(f"{label}: {path}:{line} {target} ({count})"
                        for (path, line, target), count in sorted(difference.items()))
    print(f"{len(files)} Python files; {sum(launches.values())} direct launches; "
          f"{sum(embedded.values())} embedded-script launches; {sum(documented.values())} exceptions")
    if problems:
        raise SystemExit("\n".join(problems))
    print("Every remaining launch has exactly one exception entry.")


if __name__ == "__main__":
    main()
