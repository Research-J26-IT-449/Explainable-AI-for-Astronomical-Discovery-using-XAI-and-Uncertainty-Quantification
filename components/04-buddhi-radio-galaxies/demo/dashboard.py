"""Presentation-only RadioMorph assets; never loads models, data or results."""

from html import escape
import math


NAVIGATION = """
<nav class="rm-navigation" aria-label="Main navigation">
  <a class="rm-brand" href="#overview" aria-label="RadioMorph AI overview">
    <span class="rm-logo" aria-hidden="true">R<span>•</span></span>
    <span>RadioMorph <b>AI</b><small>RESEARCH WORKSPACE</small></span>
  </a>
  <p class="rm-nav-label">WORKSPACE</p>
  <a class="rm-nav-link is-active" href="#overview" aria-current="location"><span aria-hidden="true">◈</span> Overview</a>
  <a class="rm-nav-link" href="#classification"><span aria-hidden="true">⊞</span> Galaxy Classification</a>
  <a class="rm-nav-link" href="#explainable-ai"><span aria-hidden="true">◎</span> Explainable AI</a>
  <p class="rm-nav-label rm-nav-secondary">RESEARCH</p>
  <a class="rm-nav-link" href="#research-insights"><span aria-hidden="true">≋</span> Research Insights</a>
  <a class="rm-nav-link" href="#about-model"><span aria-hidden="true">⌘</span> About the Model</a>
  <div class="rm-sidebar-note"><span class="rm-dot"></span> LOCAL RESEARCH DEMO
    <p>Explore a prediction.<br>Question its explanation.</p>
    <small>Research prototype · FR-I / FR-II<br>Not a survey-validated service</small>
  </div>
</nav>
"""

HEADER = """
<header class="rm-header"><div><span class="rm-eyebrow">RADIO ASTRONOMY / EXPLAINABLE AI</span>
<p>AGN-Aware Radio Source Intelligence</p></div>
<div class="rm-model-status"><span class="rm-dot"></span><span>Corrected MiraBest development ResNet<small>Precomputed results · live analysis disabled</small></span></div></header>
"""

HERO = """
<section class="rm-hero" aria-labelledby="rm-hero-title">
  <div class="rm-hero-copy"><span class="rm-kicker"><span class="rm-dot"></span> THE RADIO UNIVERSE, THROUGH A MODEL'S LENS</span>
  <h1 id="rm-hero-title">Structure meets<br><em>intelligence.</em></h1>
  <p>A closer look at radio-galaxy morphology. Explore saved corrected-model classification results and the limitations of gradient-based explanations.</p>
  <a class="rm-cta" href="#research-insights">Explore research results <span aria-hidden="true">↗</span></a>
  <span class="rm-hero-footnote">A research prototype. An explanation is a starting point, not proof.</span></div>
  <figure class="rm-orbital" aria-label="Decorative abstract orbital illustration, not an observation or model output">
    <div class="rm-orbital-glow"></div>
    <svg viewBox="0 0 420 340" role="img" aria-labelledby="orbital-title">
      <title id="orbital-title">Abstract radio-wave orbits — decorative illustration</title>
      <defs><radialGradient id="rm-core"><stop stop-color="#efffff"/><stop offset=".3" stop-color="#7de7ee"/><stop offset="1" stop-color="#6d6ff0" stop-opacity="0"/></radialGradient>
      <linearGradient id="rm-orbit"><stop stop-color="#59dbeb"/><stop offset=".5" stop-color="#8580ef"/><stop offset="1" stop-color="#c8b8ff" stop-opacity=".15"/></linearGradient></defs>
      <g class="rm-orbit-lines" fill="none" stroke="url(#rm-orbit)">
        <ellipse cx="210" cy="165" rx="162" ry="51" transform="rotate(-30 210 165)"/>
        <ellipse cx="210" cy="165" rx="132" ry="79" transform="rotate(42 210 165)"/>
        <ellipse cx="210" cy="165" rx="99" ry="145" transform="rotate(-36 210 165)"/>
        <ellipse cx="210" cy="165" rx="66" ry="115" transform="rotate(65 210 165)"/>
        <path d="M34 165h352M210 24v282" stroke-dasharray="2 8" stroke-opacity=".25"/>
      </g>
      <circle cx="210" cy="165" r="51" fill="url(#rm-core)"/>
      <circle class="rm-orbit-pulse" cx="210" cy="165" r="16" fill="none" stroke="#b9f4ff" stroke-opacity=".65"/>
      <circle cx="72" cy="232" r="5" fill="#81e3e9"/><circle cx="316" cy="210" r="4" fill="#b4a5ff"/>
      <circle cx="187" cy="33" r="3" fill="#93b9eb"/>
    </svg><figcaption>ABSTRACT ORBITAL STUDY / DECORATIVE</figcaption>
  </figure>
</section>
<div class="rm-workflow" aria-label="Research presentation workflow"><span><b>01</b> Read validation results</span><i aria-hidden="true">→</i><span><b>02</b> Compare attribution</span><i aria-hidden="true">→</i><span><b>03</b> Review limitations</span></div>
"""

INSIGHTS = """
<div class="rm-section-heading"><span class="rm-eyebrow">RESEARCH CONTEXT</span><h2>Research Insights</h2><p>What this demonstration can tell you. And what it cannot.</p></div>
<div class="rm-info-grid">
 <article class="rm-info-card"><span class="rm-card-icon">01 / MORPHOLOGY</span><h3>Two structural classes</h3><p><b>FR-I</b> is associated with edge-darkened radio structure; <b>FR-II</b> with edge-brightened structure. This model predicts dataset classes, not a verified physical source identity.</p></article>
 <article class="rm-info-card"><span class="rm-card-icon">02 / DATASET</span><h3>MiraBest context</h3><p>The selected CNN was developed on MiraBest grayscale radio images. A compatible upload is not proof of survey provenance. External and web-image performance remains unverified.</p></article>
 <article class="rm-info-card"><span class="rm-card-icon">03 / EXPLANATIONS</span><h3>Attribution, not evidence</h3><p>Grad-CAM uses gradients of a class logit at the final residual block. Its display heatmap is normalized per image; colors cannot be compared as absolute attribution across sources.</p></article>
</div>
<aside class="rm-integrity-note"><span class="rm-eyebrow">KNOWN RESEARCH LIMITATION</span><h3>Development validation is not independent generalization.</h3><p>The corrected MiraBest development cohort quarantines five exact-content records with conflicting labels. Validation: <b>87.07% accuracy</b>, <b>0.8644 macro F1</b> on 116 images; the checkpoint was selected using this validation cohort. External generalization and independent TEST performance are not established.</p><p>Corrected Grad-CAM evaluation covered 116 images; broader secondary analysis remains <b>incomplete</b>. Noise tests showed prediction instability and degenerate transformed maps. Expert-mask physics alignment and corrected-model OOD evaluation are unavailable; parameter-randomization sanity checks remain deferred. No physical causation or direct black-hole detection is claimed.</p></aside>
"""

ABOUT = """
<div class="rm-section-heading"><span class="rm-eyebrow">UNDER THE HOOD</span><h2>About the Model</h2><p>A focused research workflow, with explicit boundaries.</p></div>
<div class="rm-about-grid"><article class="rm-info-card"><h3>A lightweight residual CNN</h3><p>Single-channel input. A 32-channel stem and three residual blocks with widths 32 / 64 / 128. Adaptive 2×2 pooling feeds a 512 → 64 → 2 classifier.</p><div class="rm-tags"><span>FR-I = 0</span><span>FR-II = 1</span><span>Evaluation mode</span></div></article>
<article class="rm-info-card"><h3>One upload. Two perspectives.</h3><p>The unchanged input guard and checkpoint-matched frozen OOD checks precede classification. Corrected-model OOD calibration is currently unavailable, so uploads fail closed; historical calibration is never reused. AUTO / WEB compatibility is also unavailable for the corrected model.</p><p>Accepted single-representation inputs can receive a separate gradient-enabled Grad-CAM pass on a private model copy. Target: original predicted class. Layer: <code>blocks.2.convolution.3</code>. Native map: 19×19; bilinear display: 150×150. Grad-CAM is explanatory visualization, not proof of physical causation. Softmax confidence is uncalibrated; model weights and thresholds are not changed.</p></article></div>
<div class="rm-footer"><span>RadioMorph AI <b>/</b> AGN-Aware Radio Source Structure Classification</span><span>J26-IT-449 · Research prototype</span></div>
"""

WAITING = """<div class="rm-result-empty"><span class="rm-result-orbit" aria-hidden="true">◎</span><span class="rm-eyebrow">LIVE ANALYSIS DISABLED</span><h3>Explore the recorded research.</h3><p>No real-time predictions are offered. Corrected-model OOD validation is required before live analysis can be enabled.</p></div>"""
RUNNING = """<div class="rm-result-empty" role="status" aria-live="polite"><span class="rm-result-orbit" aria-hidden="true">◎</span><span class="rm-eyebrow">LIVE ANALYSIS DISABLED</span><h3>Results-only research presentation.</h3><p>No real-time model prediction is offered. Review the saved research results and limitations below.</p></div>"""


def prediction_card(scores):
    """Render only callback-provided probabilities; escape any status text."""
    if scores is None or scores == {}:
        return WAITING
    if not isinstance(scores, dict):
        return ('<div class="rm-result-empty rm-result-warning" role="status">'
                '<span class="rm-eyebrow">PREDICTION STATUS</span><h3>No class prediction available</h3>'
                f'<p>{escape(str(scores))}</p><p>No class probabilities or explanation are fabricated.</p></div>')
    if (set(scores) != {"FR-I", "FR-II"}
            or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                   for v in scores.values())):
        return prediction_card("Invalid prediction display data. No probabilities displayed.")
    winner = max(scores, key=scores.get)
    bars = "".join(
        f'<div class="rm-probability"><div><b>{label}</b><span>{scores[label]:.1%}</span></div>'
        f'<progress max="1" value="{scores[label]}" aria-label="{label} probability"></progress></div>'
        for label in ("FR-I", "FR-II"))
    return (f'<div class="rm-result" role="status" aria-live="polite"><span class="rm-eyebrow">MODEL PREDICTION</span>'
            f'<div class="rm-result-head"><h3>{winner}</h3><span>{scores[winner]:.1%}<small>model confidence</small></span></div>'
            f'{bars}<p>Softmax probabilities from the selected model, not a guarantee of correctness or calibrated uncertainty.</p></div>')


NAVIGATION_JS = """() => {
  const links = [...document.querySelectorAll('.rm-nav-link')];
  const sections = links.map(link => document.querySelector(link.getAttribute('href'))).filter(Boolean);
  if (!sections.length) return;
  const updateActive = () => {
    const line = Math.min(180, innerHeight * .2);
    const atEnd = innerHeight + scrollY >= document.documentElement.scrollHeight - 3;
    const current = atEnd ? sections.at(-1) :
      ([...sections].reverse().find(section => section.getBoundingClientRect().top <= line) || sections[0]);
    for (const link of links) {
      const active = link.getAttribute('href') === '#' + current.id;
      link.classList.toggle('is-active', active);
      if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
    }
  };
  const observer = new IntersectionObserver(updateActive, {threshold: [0, .1, .5]});
  sections.forEach(section => observer.observe(section));
  let queued = false;
  window.addEventListener('scroll', () => {
    if (!queued) requestAnimationFrame(() => { queued = false; updateActive(); });
    queued = true;
  }, {passive: true});
  document.querySelectorAll('.rm-nav-link, .rm-cta, .rm-brand').forEach(link => {
    link.addEventListener('click', event => {
      const target = document.querySelector(link.getAttribute('href'));
      if (!target) return;
      event.preventDefault();
      const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
      target.scrollIntoView({behavior: reduced ? 'instant' : 'smooth', block: 'start'});
      history.replaceState(null, '', link.getAttribute('href'));
    });
  });
  updateActive();
}"""


# Gradio scopes selectors nested inside CSS media rules to descendants of its
# container. Container/document rules must live in the supported launch head.
GLOBAL_LAYOUT_HEAD = """<style id="rm-global-layout">
@media (min-width:1600px) { html body .gradio-container { padding-right:50px!important; padding-left:285px!important; } }
@media (max-width:1050px) { html body .gradio-container { padding:18px 22px 25px 220px!important; } }
@media (max-width:760px) { html body .gradio-container { padding:12px 14px 22px!important; } }
@media (prefers-reduced-motion:reduce) {
  html { scroll-behavior:auto!important; }
  .gradio-container *, .gradio-container *::before, .gradio-container *::after {
    animation:none!important; transition:none!important;
  }
}
</style>"""
