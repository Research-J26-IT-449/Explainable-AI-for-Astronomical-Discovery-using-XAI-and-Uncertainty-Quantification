"""Pure frontend checks: no model loading, dataset access or research execution."""

from pathlib import Path
import re

import pytest

from demo import app, dashboard


@pytest.fixture(autouse=True)
def prevent_inference(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    def forbidden(*a, **k):
        raise AssertionError("Research/model access is not allowed in frontend checks")
    monkeypatch.setattr(app, "predict_image", forbidden)
    monkeypatch.setattr(app, "load_calibrated_ood_detector", forbidden)
    monkeypatch.setattr(app, "generate_gradcam", forbidden)


def test_presentation_wrapper_keeps_seven_original_outputs(monkeypatch):
    original = (object(), {"FR-I": 0.73, "FR-II": 0.27}, "Handling", object(),
                object(), object(), "Explanation")
    calls = []
    monkeypatch.setattr(app, "predict_for_ui_with_gradcam",
                        lambda image, mode: calls.append((image, mode)) or original)
    image = object()
    result = app.analyze_for_dashboard(image, "external_radio")
    assert calls == [(image, "external_radio")]
    assert all(a is b for a, b in zip(result[:7], original))
    assert "73.0%" in result[7] and "27.0%" in result[7]
    assert 'value="0.73"' in result[7]
    assert "not a guarantee" in result[7]


@pytest.mark.parametrize("value", [None, {}])
def test_empty_prediction_does_not_invent_values(value):
    card = dashboard.prediction_card(value)
    assert "LIVE ANALYSIS DISABLED" in card
    assert "<progress" not in card and "%" not in card


def test_status_is_escaped_and_no_bars_for_rejected_input():
    card = dashboard.prediction_card('UNSUPPORTED: <script>alert("test")</script>')
    assert "<script>" not in card and "&lt;script&gt;" in card
    assert "<progress" not in card


@pytest.mark.parametrize("scores", [
    {"FR-I": float("nan"), "FR-II": 0.2},
    {"FR-I": float("inf"), "FR-II": 0.2},
    {"FR-I": -0.1, "FR-II": 1.1}, {"invented": 0.5},
])
def test_invalid_display_values_fail_closed(scores):
    card = dashboard.prediction_card(scores)
    assert "Invalid prediction" in card and "<progress" not in card


def test_fr_ii_prediction_comes_from_actual_input_probabilities():
    card = dashboard.prediction_card({"FR-I": 0.12, "FR-II": 0.88})
    assert "<h3>FR-II</h3>" in card and "88.0%" in card
    assert 'value="0.88"' in card


def test_layout_has_working_navigation_and_analysis_wiring():
    demo = app.create_demo()
    try:
        config = demo.get_config_file()
        components = {c["id"]: c for c in config["components"]}
        ids = {c["props"].get("elem_id") for c in components.values()}
        anchors = re.findall('href="#([^\"]+)"', dashboard.NAVIGATION)
        assert set(anchors) <= ids
        assert set(anchors) == {"overview", "classification", "explainable-ai", "research-insights", "about-model"}
        modes = next(c for c in components.values() if c["props"].get("elem_id") == "rm-mode")
        assert {pair[1] for pair in modes["props"]["choices"]} == {"mirabest", "external_radio", "auto_web"}
        upload = next(c for c in components.values() if c["props"].get("elem_id") == "rm-upload")
        assert upload["props"]["sources"] == ["upload"]
        assert upload["props"]["type"] == "pil"
        assert upload["props"].get("image_mode") is None
        assert any(c["props"].get("elem_id") == "rm-analyze" for c in components.values())
        event = next(v for v in demo.fns.values() if v.fn is app.analyze_for_dashboard)
        assert len(event.outputs) == 8
        assert [c._id for c in event.inputs] == [upload["id"], modes["id"]]
        # Three real visualization components remain wired to the original positions.
        assert [o.label for o in event.outputs[3:6]] == [
            "Exact classifier input (150×150 grayscale)",
            "Grad-CAM heatmap (normalized [0,1])", "Grad-CAM overlay on classifier input"]
        assert all(o.interactive is False for o in event.outputs[3:6])
        values = " ".join(str(c["props"].get("value", "")) for c in components.values())
        assert app.XAI_NOTE in values
        assert "corrected MiraBest development cohort" in values and "incomplete" in values
        assert "conflicting labels" in values and "not independent generalization" in values
        assert "not an observation or model output" in values
        assert not demo.is_running
    finally:
        demo.close()


def test_launch_options_use_local_assets_without_network_fonts():
    options = app.dashboard_launch_options()
    assert options["share"] is False and options["server_name"] == "127.0.0.1"
    assert "IntersectionObserver" in options["js"]
    css = options["css"]
    assert css == Path(app.__file__).with_name("dashboard.css").read_text(encoding="utf-8")
    assert "@import" not in css and "https://" not in css and "url(" not in css
    assert css.count("{") == css.count("}")
    assert "prefers-reduced-motion" in css and ":focus-visible" in css
    assert "max-width: 760px" in css and "max-width: 1050px" in css
    assert "object-fit: contain" in css
    assert not options["theme"]._font_css
    assert "rm-global-layout" in options["head"]
    assert "max-width:760px" in options["head"]
    assert "prefers-reduced-motion" in options["head"]


def test_idle_and_error_states_remain_explicit_without_inference():
    result = app.analyze_for_dashboard(None)
    assert result[3:6] == (None, None, None)
    assert "LIVE ANALYSIS DISABLED" in result[7]
    assert "Please upload" in result[2]
    result = app.analyze_for_dashboard(object())
    assert result[1] == "UNSUPPORTED"
    assert result[3:6] == (None, None, None)
    assert "No class prediction available" in result[7]
