"""Tests for the Phase 9 Streamlit app: every page renders headless, without errors.

Uses Streamlit's AppTest on the saved artifacts (skipped if the app assets are missing).
The whole app runs once through ``st.navigation`` (its default page); every page function
also runs on its own, with its main controls exercised.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ehrgraph.config import load_config

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
TIMEOUT = 300  # seconds per run; the first loads the Predictor
PAGE_SCRIPT = """
import sys
sys.path.insert(0, {app_dir!r})
import streamlit_app as app
app.{page}()
"""
PAGES = ["overview_page", "new_patient_page", "explorer_page", "network_page", "graph_page",
         "results_page"]


@pytest.fixture(scope="module", autouse=True)
def assets_present() -> None:
    if not load_config().paths.app_stats_json.exists():
        pytest.skip("app assets missing: run scripts/09_app_assets.py")


def _page(name: str) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT.format(app_dir=str(APP.parent), page=name),
                             default_timeout=TIMEOUT)
    at.run()
    return at


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_the_app_runs_through_its_navigation() -> None:
    at = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    at.run()
    _assert_clean(at)
    assert at.title[0].value.startswith("Electronic health record networks")
    assert any("Not for clinical use" in w.value for w in at.sidebar.warning)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page: str) -> None:
    _assert_clean(_page(page))


def test_new_patient_page_with_every_example() -> None:
    cfg = load_config()
    at = _page("new_patient_page")
    assert any("Not for clinical use" in w.value for w in at.warning)  # permanent disclaimer
    for demo in cfg.inference.demo_patients:
        at.selectbox(key="np_example").select(demo.name).run()
        at.button(key="np_load").click().run()
        _assert_clean(at)
        assert any(m.value.startswith("**At risk for") for m in at.markdown)
        # The supervised probability comes last, with its caveat.
        assert "near chance, not for decisions" in at.caption[-1].value
        at.toggle(key="np_band").set_value(True).run()
        _assert_clean(at)
        at.toggle(key="np_band").set_value(False).run()


def test_interactive_pages_accept_their_controls() -> None:
    at = _page("explorer_page")
    at.selectbox(key="ex_phenotype").select(2).run()
    at.toggle(key="ex_band").set_value(True).run()
    _assert_clean(at)

    at = _page("network_page")
    at.slider(key="net_lift").set_value(float(at.slider(key="net_lift").max)).run()
    at.radio(key="net_weight").set_value("jaccard").run()
    _assert_clean(at)

    at = _page("graph_page")
    at.select_slider(key="g_year").set_value(load_config().trajectories.start_year).run()
    at.radio(key="g_positions").set_value("year").run()
    at.radio(key="g_edges").set_value("new").run()
    at.selectbox(key="g_highlight").select(4).run()
    _assert_clean(at)
