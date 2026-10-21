"""Thin Gradio UI for the selected MiraBest ResNet classifier."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from demo.research_results import public_mode


def predict_image(image, *, input_mode="mirabest"):
	if public_mode():
		return {"predicted_class": "UNSUPPORTED", "confidence": None, "probabilities": {},
			"message": "Live prediction is disabled. This public demo presents precomputed research results; "
			"corrected-model OOD validation is required before live uploads can be enabled.", "ood": None}
	from src.inference.corrected_demo import predict_image as predict
	return predict(image, input_mode=input_mode)


def load_calibrated_ood_detector():
	if public_mode():
		raise PermissionError("Live public inference is disabled.")
	from src.inference.corrected_demo import load_calibrated_ood_detector as load
	return load()


def prepare_guarded_image(image):
	from src.ood.input_guard import prepare_guarded_image as prepare
	return prepare(image)


def prepare_external_radio_image(image):
	from src.ood.external_radio import prepare_external_radio_image as prepare
	return prepare(image)


def generate_gradcam(*args, **kwargs):
	from src.xai.gradcam import generate_gradcam as generate
	return generate(*args, **kwargs)


EXPLANATION_LOCK = RLock()
XAI_NOTE = (
    "Interactive demonstration only — NOT a scientifically validated Phase 3B evaluation result. "
    "Grad-CAM highlights regions contributing positively to the predicted class logit; "
    "it does not identify verified jets, lobes or other astronomical structures, "
    "and does not prove correctness or faithfulness."
)


DEMO_NOTE = (
	"**MiraBest / Supported Radio Image:** Corrected development-model path for grayscale radio "
	"morphology images.\n\n"
	"**External radio map:** Experimental path for compatible grayscale or achromatic external "
	"radio maps.\n\n"
	"**AUTO / WEB RADIO IMAGE:** Experimental compatibility path for some web/survey radio "
	"images. Unsupported or out-of-domain images are rejected rather than forced into FR-I or FR-II.\n\n"
	"**Scientific limitation:** External-image classification is experimental and has not been "
	"validated across arbitrary surveys or web images."
)


def predict_for_ui(image: Any, input_mode: str = "mirabest") -> tuple[Any, dict[str, float] | str]:
	"""Adapt reusable inference output to Gradio image and label components."""

	result = predict_image(image) if input_mode == "mirabest" else predict_image(image, input_mode=input_mode)
	if result["predicted_class"] in {"UNSUPPORTED", "AMBIGUOUS"}:
		return image, result["predicted_class"] + ": " + result["message"]
	return image, result["probabilities"]


def predict_for_ui_with_details(image: Any, input_mode: str = "mirabest"):
	"""Expose useful handling and scientific context without numeric gate details."""
	result = predict_image(image, input_mode=input_mode)
	return _prediction_display(image, input_mode, result)


def _prediction_display(image, input_mode, result):
	"""Keep classification presentation identical for both UI adapters."""
	label = result["predicted_class"]
	if label in {"UNSUPPORTED", "AMBIGUOUS"}:
		return image, label + ": " + result["message"], result["message"]
	display = {"FRI": "FR-I", "FRII": "FR-II"}
	scores = {display.get(key, key): value for key, value in result["probabilities"].items()}
	handling = result.get("input_handling", input_mode)
	representations = ", ".join(result.get("representations", []))
	note = f"{display.get(label, label)} | confidence: {result['confidence']:.1%} | handling: {handling}. "
	if representations:
		note += f"Representation(s): {representations}. "
	note += result["message"]
	return image, scores, note


def predict_for_ui_with_gradcam(image: Any, input_mode: str = "mirabest"):
	"""Explain only accepted uploads, on a private model and the exact guarded input.

	No research dataset or scientific evaluation runner is used. The lock serializes
	this callback's cached-loader access; Grad-CAM hooks never touch a cached model.
	Every return clears previous heatmaps, including rejection/error paths.
	"""
	if image is None:
		return None, None, "Please upload one radio galaxy image.", None, None, None, XAI_NOTE
	try:
		with EXPLANATION_LOCK:
			result = predict_image(image, input_mode=input_mode)
			preview, scores, note = _prediction_display(image, input_mode, result)
			base = (preview, scores, note)
			if result["predicted_class"] not in {"FRI", "FRII"}:
				return *base, None, None, None, "Grad-CAM unavailable: input was not accepted. " + XAI_NOTE
			if input_mode == "auto_web":
				return (*base, None, None, None,
					"Grad-CAM unavailable for AUTO / WEB combined representations. "
					"For a single compatible grayscale map select MiraBest or External radio map. " + XAI_NOTE)
			try:
				import numpy as np
				import torch
				from PIL import Image
				from matplotlib import colormaps

				prepare = {"mirabest": prepare_guarded_image,
					"external_radio": prepare_external_radio_image}[input_mode]
				guarded = prepare(image)
				# Revalidate corrected checkpoint and matching calibration; no historical fallback.
				detector = load_calibrated_ood_detector()
				if not detector.validate_tensor(guarded.tensor)["accepted"]:
					raise ValueError("Frozen OOD validation did not accept the explanation input.")
				model = deepcopy(detector.model).eval()
				with torch.no_grad():
					baseline = model(guarded.tensor).detach().clone()
				if baseline.shape != (1, 2) or not torch.isfinite(baseline).all():
					raise ValueError("Classifier logits are invalid.")
				probabilities = baseline.softmax(1)[0]
				expected = torch.tensor([result["probabilities"][name] for name in ("FRI", "FRII")])
				target = ("FRI", "FRII").index(result["predicted_class"])
				if (int(baseline.argmax(1)) != target
						or not torch.allclose(probabilities.cpu(), expected, atol=1e-6, rtol=1e-5)):
					raise ValueError("Explanation prediction differs from the displayed classification.")
				cam = generate_gradcam(model, guarded.tensor, target_class=target)
				if (cam.target_class != target or cam.predicted_class != target
						or not torch.equal(cam.logits, baseline)):
					raise ValueError("Grad-CAM logits or target differ from the classification.")
				if cam.status == "degenerate":
					return (*base, guarded.image, None, None,
						"Degenerate Grad-CAM: no spatially informative explanation is available. " + XAI_NOTE)
				heat = cam.heatmap.detach().cpu().numpy()
				if (cam.status != "valid" or heat.shape != (150, 150)
						or not np.isfinite(heat).all() or heat.min() < 0 or heat.max() > 1):
					raise ValueError("Grad-CAM heatmap is invalid.")
				rgba = colormaps["inferno"](heat, bytes=True)
				heatmap = Image.fromarray(rgba[..., :3])
				rgba[..., 3] = np.rint(heat * 0.5 * 255).astype(np.uint8)
				overlay = Image.alpha_composite(guarded.image.convert("RGBA"), Image.fromarray(rgba))
				return (*base, guarded.image, heatmap, overlay,
					f"Predicted-class explanation: {'FR-I' if target == 0 else 'FR-II'}. "
					f"Layer: {cam.layer_name}. Heatmap: 150×150, normalized [0,1]; "
					"brighter colors indicate higher relative attribution. " + XAI_NOTE)
			except Exception as error:
				return (*base, None, None, None, f"Grad-CAM unavailable: {error}. " + XAI_NOTE)
	except Exception as error:
		return image, "UNSUPPORTED", f"Prediction unavailable: {error}", None, None, None, XAI_NOTE



def analyze_for_dashboard(image: Any, input_mode: str = "mirabest"):
	"""Append presentation HTML; the existing seven scientific outputs are unchanged."""
	from demo.dashboard import prediction_card
	result = predict_for_ui_with_gradcam(image, input_mode)
	return (*result, prediction_card(result[1]))


def dashboard_launch_options():
	"""Gradio 6 accepts CSS/theme/JS on launch, not the Blocks constructor."""
	import gradio as gr
	from demo.dashboard import NAVIGATION_JS, GLOBAL_LAYOUT_HEAD
	return {
		"css": (Path(__file__).with_name("dashboard.css")).read_text(encoding="utf-8"),
		"js": NAVIGATION_JS,
		"head": GLOBAL_LAYOUT_HEAD,
		"theme": gr.themes.Base(primary_hue="cyan", secondary_hue="violet", neutral_hue="slate",
			font=["Segoe UI", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"]),
		"footer_links": [],
		"share": False,
		"server_name": "127.0.0.1",
	}


def create_demo():
	"""Build the local research dashboard; no model or dataset access at startup."""
	try:
		import gradio as gr
	except ImportError as error:
		raise RuntimeError("Gradio is not installed. Run: pip install gradio") from error
	from demo import dashboard as ui

	with gr.Blocks(title="RadioMorph AI | Research Workspace", analytics_enabled=False,
		fill_width=True) as demo:
		gr.HTML(ui.NAVIGATION, elem_id="rm-navigation")
		with gr.Column(elem_id="overview", min_width=0):
			gr.HTML(ui.HEADER)
			gr.HTML(ui.HERO)
		with gr.Column(elem_id="classification", min_width=0):
			gr.HTML('<div class="rm-section-heading"><span class="rm-eyebrow">01 / CLASSIFICATION WORKSPACE</span>'
				'<h2>Live prediction — currently unavailable</h2><p>Explore the precomputed research results below. Upload analysis is disabled pending corrected-model OOD validation.</p></div>')
			with gr.Row(equal_height=True):
				with gr.Column(scale=6, min_width=280, elem_classes="rm-panel"):
					gr.Markdown("### Source image")
					image = gr.Image(type="pil", image_mode=None, sources=["upload"], height=300,
						label="Original uploaded image (live analysis disabled)", elem_id="rm-upload", interactive=False,
						buttons=["fullscreen"], placeholder="Live uploads disabled — explore saved research results",
						alt_text="Original user-uploaded image")
					gr.Markdown("Live uploads are disabled in this research presentation. "
						"No visitor image receives a prediction; the existing input and calibration safeguards remain intact.")
					mode = gr.Radio(choices=[("MiraBest / Supported Radio Image", "mirabest"),
						("External radio map (experimental)", "external_radio"),
						("AUTO / WEB RADIO IMAGE (experimental)", "auto_web")],
						value="mirabest", label="Input preparation (unavailable)", elem_id="rm-mode", interactive=False)
					submit = gr.Button("Live analysis unavailable", variant="primary", elem_id="rm-analyze", interactive=False)
				with gr.Column(scale=5, min_width=280, elem_classes="rm-panel"):
					gr.Markdown("### Classification insight")
					result_card = gr.HTML(ui.WAITING, elem_id="rm-result-card")
					# Retain the existing output contract; HTML is presentation only.
					preview = gr.Image(type="pil", label="Uploaded image preview", interactive=False, visible=False)
					prediction = gr.Label(num_top_classes=2, label="FR-I / FR-II prediction or unsupported status", visible=False)
					details = gr.Textbox(label="Input handling and scientific note", lines=3,
						placeholder="Live analysis unavailable. No class scores are fabricated.",
						interactive=False, elem_classes="rm-status")
					gr.HTML('<div class="rm-explanation-note"><p><b>Research use only.</b> '
						'Corrected-model OOD calibration is unavailable: upload prediction and explanation '
						'remain disabled. Historical calibration is not a substitute.</p></div>')
		with gr.Column(elem_id="explainable-ai", min_width=0):
			gr.HTML('<div class="rm-section-heading"><span class="rm-eyebrow">02 / ATTRIBUTION STUDIO</span>'
				'<h2>Explainable AI (Grad-CAM)</h2><p>Live generation is disabled. Saved explanation coverage and example availability are documented in Research Results below.</p></div>')
			with gr.Row(equal_height=True):
				with gr.Column(min_width=240, elem_classes=["rm-panel", "rm-visual-card"]):
					gr.HTML('<div class="rm-card-index">A / MODEL VIEW</div><h3>Exact CNN Input</h3>')
					prepared = gr.Image(type="pil", label="Exact classifier input (150×150 grayscale)",
						height=250, interactive=False, buttons=["fullscreen"], elem_classes="rm-scientific-image")
					gr.HTML('<p class="rm-caption">Actual 150×150 grayscale image used by the classifier. '
						'Existing preprocessing only; displayed larger without altering inference pixels.</p>')
				with gr.Column(min_width=240, elem_classes=["rm-panel", "rm-visual-card"]):
					gr.HTML('<div class="rm-card-index">B / CLASS ATTRIBUTION</div><h3>Grad-CAM Heatmap</h3>')
					heatmap = gr.Image(type="pil", label="Grad-CAM heatmap (normalized [0,1])",
						height=250, interactive=False, buttons=["fullscreen"], elem_classes="rm-scientific-image")
					gr.HTML('<div class="rm-legend" aria-label="Inferno color scale, low 0 to high 1"><span>0 · low</span>'
						'<span class="rm-legend-bar"></span><span>1 · high</span></div>'
						'<p class="rm-caption">Positive predicted-class attribution, normalized within this image. '
						'This is not a probability map or physical radio intensity.</p>')
				with gr.Column(min_width=240, elem_classes=["rm-panel", "rm-visual-card"]):
					gr.HTML('<div class="rm-card-index">C / SPATIAL CONTEXT</div><h3>Grad-CAM Overlay</h3>')
					overlay = gr.Image(type="pil", label="Grad-CAM overlay on classifier input",
						height=250, interactive=False, buttons=["fullscreen"], elem_classes="rm-scientific-image")
					gr.HTML('<p class="rm-caption">Transparent attribution over the exact model input. '
						'Highlighted regions are model contributions, not verified astronomical structures.</p>')
			xai_status = gr.Textbox(label="Explanation status and interpretation", lines=3,
				placeholder="Live Grad-CAM unavailable. Explore precomputed research results below.",
				interactive=False, elem_classes="rm-status")
			gr.HTML('<div class="rm-explanation-note"><p>' + ui.escape(XAI_NOTE) + '</p>'
				'<p>AUTO / WEB combined representations do not have a single Grad-CAM preview. '
				'Rejected inputs, failed explanations and degenerate maps remain explicitly unavailable.</p></div>')
		with gr.Column(elem_id="research-insights", min_width=0):
			from demo import research_results
			gr.HTML(research_results.research_html())
			gr.Image(value=str(research_results.ASSETS / "confusion_matrix.png"),
				label="Saved corrected validation confusion matrix — 116 images, not TEST", interactive=False,
				height=400, buttons=["fullscreen"])
			examples = research_results.local_examples()
			if examples:
				gr.Gallery(value=examples, label="Precomputed Grad-CAM — private local research view only",
					columns=2, interactive=False)
			else:
				gr.Markdown("**Precomputed Grad-CAM examples:** 116 explanations were generated. "
					"Source-image overlays are excluded from the public package because redistribution "
					"rights have not been established. No replacement or fabricated example is shown.")
			gr.HTML(ui.INSIGHTS)
		with gr.Column(elem_id="about-model", min_width=0):
			gr.HTML(ui.ABOUT)

		outputs = [preview, prediction, details, prepared, heatmap, overlay, xai_status, result_card]
		begin = submit.click(lambda: ui.RUNNING, outputs=result_card, queue=False)
		analysis = begin.then(analyze_for_dashboard, inputs=[image, mode], outputs=outputs,
			concurrency_limit=1)
		def clear_outputs():
			return None, None, "", None, None, None, "Live analysis is disabled; saved research results are available below.", ui.WAITING
		# Clear stale results and request cancellation of pending analysis on changes.
		image.change(clear_outputs, outputs=outputs, cancels=[analysis], queue=False)
		mode.change(clear_outputs, outputs=outputs, cancels=[analysis], queue=False)
	return demo


if __name__ == "__main__":
	create_demo().launch(**dashboard_launch_options())
