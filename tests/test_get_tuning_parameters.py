import os
import sys

# ensure project root is on sys.path so `scripts` package imports work during pytest
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.get_tuning_parameters import parse_tuning_parameters


def test_parse_simple(tmp_path):
    src = tmp_path / "simple.cl"
    src.write_text(
        """
#define TYPE_T float
// a comment
#if OCL_DIM_L_1 == 2
  // do something
#endif
// usage of tuning params
int arr[NUM_WG_L_1 * NUM_WI_L_1];
"""
    )

    res = parse_tuning_parameters(str(src))
    names = set(res["parameters"].keys())
    assert "OCL_DIM_L_1" in names
    assert "NUM_WG_L_1" in names
    assert "NUM_WI_L_1" in names
    assert "TYPE_T" not in names


def test_parse_template_file():
    # ensure we can parse one of the real kernel templates
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, "kernel-template", "gemm_1.cl")
    assert os.path.exists(path)
    res = parse_tuning_parameters(path)
    names = set(res["parameters"].keys())
    # common tuning parameters should appear
    assert any(n.startswith("NUM_WG_") or n.startswith("NUM_WI_") for n in names)


def test_parse_all_templates():
    base = os.path.dirname(os.path.dirname(__file__))
    td = os.path.join(base, "kernel-template")
    assert os.path.isdir(td)
    for fn in os.listdir(td):
        if not fn.endswith(".cl"):
            continue
        path = os.path.join(td, fn)
        res = parse_tuning_parameters(path)
        assert isinstance(res, dict)
        assert "parameters" in res
