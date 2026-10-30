"""Exercise the required-CUDA policy using real nested pytest sessions."""

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def cuda_policy(pytester: pytest.Pytester) -> pytest.Pytester:
    pytester.makeconftest(
        """
        from runtimes.onnx.tests.conftest import (
            pytest_addoption,
            pytest_collection_finish,
            pytest_runtest_makereport,
        )
        """
    )
    pytester.makeini("[pytest]\nmarkers = cuda: requires a CUDA device")
    return pytester


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("case", ["pass", "skip", "setup-skip", "empty"])
def test_required_cuda_coverage(
    cuda_policy: pytest.Pytester, required: bool, case: str
) -> None:
    marker = "@pytest.mark.cuda" if case != "empty" else ""
    if case == "setup-skip":
        marker += '\n@pytest.mark.skip(reason="no CUDA device")'
    body = 'pytest.skip("no CUDA device")' if case == "skip" else "pass"
    cuda_policy.makepyfile(f"import pytest\n{marker}\ndef test_device():\n    {body}\n")
    args = ["--require-cuda"] if required else []
    result = cuda_policy.runpytest_subprocess(*args)
    if required and case == "empty":
        assert result.ret != 0
        result.stderr.fnmatch_lines(["*requires at least one CUDA test*"])
    elif required and case in ("skip", "setup-skip"):
        result.assert_outcomes(
            **({"errors": 1} if case == "setup-skip" else {"failed": 1})
        )
        assert result.ret != 0
        result.stdout.fnmatch_lines(["*CUDA test skipped with --require-cuda*"])
    elif case in ("skip", "setup-skip"):
        result.assert_outcomes(skipped=1)
        assert result.ret == 0
    else:
        result.assert_outcomes(passed=1)
        assert result.ret == 0
