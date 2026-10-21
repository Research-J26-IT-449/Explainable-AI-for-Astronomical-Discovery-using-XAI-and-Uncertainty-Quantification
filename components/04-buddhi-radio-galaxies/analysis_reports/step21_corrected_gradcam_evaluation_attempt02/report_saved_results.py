"""Audit/report completed corrected XAI records only; no image decode or model call."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
D=Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def format_value(value):
    return 'unavailable' if value is None else f'{value:.6f}'


def main():
    completion=read(D/'complete.json')
    process=read(D/'process_finished.json')
    assert process['exit_code']==0 and not (D/'failure.json').exists()
    preflight=read(D/'preflight.json')
    protocol=read(D/'protocol.json')
    summary=read(D/'quantitative_summary.json')
    ids=protocol['validation_indices']
    records=[read(D/'records'/f'index_{i}.json') for i in ids]
    telemetry=[read(D/'telemetry'/f'index_{i}.json') for i in ids]
    assert len(records)==len(set(ids))==116
    assert Counter(r['ground_truth_label'] for r in records)=={0:51,1:65}
    assert sum(r['classification_correct'] for r in records)==101
    assert all(r['dataset_index']==i for r,i in zip(records,ids))
    assert all(r['target_class']==r['predicted_class'] and r['step20_probability_max_difference']<=1e-5 for r in records)
    assert all(sha(ROOT/name)==h for name,h in preflight['implementation_sha256'].items())
    assert all(sha(ROOT/name)==h for name,h in preflight['baseline'].items())
    assert all(sha(ROOT/name)==h for name,h in preflight['protected'].items())
    assert sha(ROOT/protocol['checkpoint_reference'])==protocol['checkpoint_sha256']
    assert sha(ROOT/protocol['manifest_reference'])==protocol['manifest_sha256']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()==preflight['git']['head']
    assert sha(ROOT/'.git/index')==preflight['git']['index_sha256']
    mismatch={p:{'snapshot':h,'final':sha(D/p)} for p,h in completion['output_sha256'].items() if sha(D/p)!=h}
    # The engine snapshots an externally growing operational stdout log before its final print.
    assert set(mismatch)<= {'execution_stdout.log'}, mismatch
    if mismatch:
        raw=(D/'execution_stdout.log').read_bytes()
        endline=b'COMPLETE: 116 corrected validation explanations; quantitative summary saved.'
        cut=raw.rfind(endline)
        assert cut>=0 and hashlib.sha256(raw[:cut]).hexdigest()==mismatch['execution_stdout.log']['snapshot']
    for r in records:
        assert len(r['stability'])==30
        assert all(v['target_class']==r['target_class'] for v in r['stability'])
        if r['faithfulness']['status']=='available':
            f=r['faithfulness']
            assert len(f['random_aopc'])==20
            assert set(f['random_aopc'])=={str(i) for i in range(1000,1020)}
            assert len(f['result']['gradcam']['deleted_logits'])==6
        with np.load(D/r['heatmap_reference'],allow_pickle=False) as maps:
            assert maps['raw_cam'].shape==(19,19)
            assert maps['upsampled_heatmap'].shape==maps['heatmap'].shape==(150,150)
            assert all(np.isfinite(maps[k]).all() for k in maps.files)
            assert maps['raw_cam'].min()>=0 and 0<=maps['heatmap'].min()<=maps['heatmap'].max()<=1
    family=defaultdict(float)
    for row in telemetry:
        for name,value in row['family_seconds'].items():
            family[name]+=value
    durations=np.array([r['elapsed_seconds'] for r in telemetry])
    peak=max(t['process_lifetime_peak_bytes'] for t in telemetry)
    maximum_sampled=max(t['maximum_sampled_working_set_bytes'] for t in telemetry)
    assert maximum_sampled<=2*1024**3 and durations.max()<=900
    negative=sum(r['faithfulness']['status']=='available' and r['faithfulness']['gradcam_aopc']['probability']<0 for r in records)
    groups=defaultdict(list)
    for record in records:
        for v in record['stability']:
            spec=v['transformation']
            name=f"{spec['kind']} {spec['value']}"
            groups[name].append(v)
    diagnostics={name:dict(comparisons=len(vs),flips=sum(v['class_flip'] for v in vs),
        unavailable=dict(Counter(v['reason'] for v in vs if v['status']!='valid')),
        mean_clipped_fraction=float(np.mean([v['clipped_fraction'] for v in vs])),
        max_clipped_fraction=max(v['clipped_fraction'] for v in vs),
        flips_by_true_class={str(label):sum(v['class_flip'] for r in records if r['ground_truth_label']==label
            for v in r['stability'] if f"{v['transformation']['kind']} {v['transformation']['value']}"==name) for label in [0,1]})
        for name,vs in groups.items()}
    with (D/'stability_diagnostics.json').open('x') as stream:
        json.dump(diagnostics,stream,indent=2)
    selected=['cam_probability_aopc30','random_probability_aopc30','brightness_probability_aopc30',
              'CAM_minus_random_probability_AOPC30','CAM_minus_brightness_probability_AOPC30']
    lines=['# Corrected-model Grad-CAM evaluation report','',
        '**Verdict: XAI PARTIALLY COMPLETED — LIMITATIONS DOCUMENTED.**',
        'All essential endpoints of Corrected-GradCAM-Validation-v1.0.0 completed for the full 116-image '
        'corrected validation cohort. This does not complete every historical Phase 3B secondary analysis.', '',
        '## Identity, authorization and verification','',
        f"Protocol: {protocol['protocol']}. Dataset protocol: {protocol['dataset_protocol']}.",
        f"Checkpoint: `{protocol['checkpoint_reference']}`; SHA-256 `{protocol['checkpoint_sha256']}`.",
        f"Manifest: `{protocol['manifest_reference']}`; SHA-256 `{protocol['manifest_sha256']}`.",
        'The separately versioned protocol changes model/cohort identity only where explicitly approved, '
        'and documents essential/secondary scope before execution. Historical protocol/YAML and artifacts '
        'were never edited. Applicable mathematical modules were reused unchanged. Historical tie-breaking '
        'namespaces remain explicit mathematical conventions, not misrepresented execution provenance.', '',
        'Command (repository root):','```powershell',
        r'.venv\Scripts\python.exe -B analysis_reports/step21_corrected_gradcam_evaluation_attempt02/evaluate_corrected_xai.py --execute',
        '```','',
        f"Preflight passed. Synthetic integration checks passed without real data/model forward calls; "
        '188 focused Grad-CAM/faithfulness/stability/morphology tests passed (0 failed, 6.02 seconds). '
        'Safe weights_only CPU checkpoint load; strict ResNetCNN architecture; eval mode; parameters frozen; '
        'original validation preprocessing unchanged. Exactly 116 validation rows decoded, one at a time '
        '(maximum one selected raw image per call, within the eight-image bound). No TRAIN/TEST/quarantined '
        'pixels decoded; archived label values skipped; labels came from the corrected manifest.', '',
        '## Genuine explanation coverage','',
        f"116/116 unique records completed; class supports 51 FR-I/65 FR-II, 101 correct/15 incorrect. "
        f"Original CAM statuses: {summary['gradcam_status_counts']}. All predictions matched Step 20; "
        f"maximum probability difference {max(r['step20_probability_max_difference'] for r in records):.9g} "
        '(tolerance 1e-5 for float32 single-image versus batch computation). No separate classification '
        'pass, model selection, retraining or tuning occurred.',
        'Target model.blocks[2].convolution[3]; original predicted-class pre-softmax logit; temporary '
        'activation hook and autograd.grad. Native nonnegative CAM 19x19; bilinear 150x150, '
        'align_corners=False; original geometry and normalization preserved. All weights/BatchNorm '
        'buffers stayed exactly equal after every image; parameter gradients remained absent. '
        'All three saved map arrays were rechecked for shapes, finiteness and valid ranges.', '',
        '## Primary probability AOPC30','',
        'Border-median replacement; native tile rankings fixed from original input; exact nested '
        '5/10/15/20/25/30% budgets; exactly 20 seeded random controls (1000–1019), averaged per image. '
        'AOPC is mean original-minus-deleted fixed-class probability; logit/margin curves are also saved. '
        'Negative values are retained. Deletion inputs may be out of distribution; no historical '
        'OOD calibration was loaded and no perturbed score was filtered as publicly supported.', '',
        '| Endpoint | Available / 116 | Mean | Median | FR-I mean | FR-II mean | Correct mean | Incorrect mean |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in selected:
        s=summary['metric_summaries'][name]
        lines.append(f"| {name} | {s['overall']['available']} | {format_value(s['overall']['mean'])} | "
            f"{format_value(s['overall']['median'])} | {format_value(s['by_class'][0]['mean'])} | "
            f"{format_value(s['by_class'][1]['mean'])} | {format_value(s['correctly_classified']['mean'])} | "
            f"{format_value(s['misclassified']['mean'])} |")
    lines+=['',f'Negative CAM probability AOPC observations: {negative}. Zero/negative deletion effects '
        'are not clipped or omitted. Positive paired contrasts favor CAM sensitivity over that control; '
        'negative contrasts do not. Neither direction proves astronomical correctness.','',
        'Paired class-stratified image bootstrap: 10,000 replicates, seed42, 95% percentile intervals. '
        'These are exploratory, conditional on metric availability and source-independence assumptions. '
        'No confirmatory p-values or significance claims were introduced.','']
    for name,b in summary['paired_aopc_bootstrap'].items():
        lines.append(f"- {name}: overall estimate {b['overall_estimates']}, overall 95% interval "
                     f"{b['overall_intervals']}; equal-class macro estimate {b['macro_estimates']}, "
                     f"macro interval {b['macro_intervals']}; coverage {b['coverage']}.")
    lines+=['','## Stability and robustness','',
        'Thirty independent predefined transformations per image, including 20 noise realizations. '
        'Fixed original class is explained even when prediction flips. Continuous maps inverse-aligned '
        'before top-k selection in common valid support; support <80% unavailable. Primary top10% '
        'Jaccard and 5%/20% sensitivities use existing tie policy. This Jaccard is explanation stability, '
        'not expert-reference astronomical localization. Noise repeats are averaged within image/level '
        'before statistical summaries; comparisons below are diagnostic counts, not independent images.','',
        '| Condition | Comparisons | Prediction flips | Unavailable comparisons | Mean clipped fraction |',
        '|---|---:|---:|---:|---:|']
    for name,values in diagnostics.items():
        lines.append(f"| {name} | {values['comparisons']} | {values['flips']} | "
                     f"{sum(values['unavailable'].values())} | {values['mean_clipped_fraction']:.6f} |")
    lines+=['','Image-level primary top10% Jaccard summaries:', '',
        '| Endpoint | Available images | Mean | FR-I mean | FR-II mean | Correct mean | Incorrect mean |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summary['metric_summaries'].items():
        if name.startswith('stability.') and name.endswith('top0.1'):
            lines.append(f"| {name} | {s['overall']['available']} | {format_value(s['overall']['mean'])} | "
                f"{format_value(s['by_class'][0]['mean'])} | {format_value(s['by_class'][1]['mean'])} | "
                f"{format_value(s['correctly_classified']['mean'])} | {format_value(s['misclassified']['mean'])} |")
    lines+=['',f"All/correct/incorrect/class stability counts and reason codes: {summary['stability_case_counts']}. "
        'Raw records retain preserving/flipped status and Jaccards separately; no flips or degeneracies '
        'were silently dropped. Noise clipping is not a physically calibrated radio-flux perturbation. '
        'Stability alone does not establish informative or faithful attribution.','',
        '## Intensity-proxy overlap and physics limitations','',
        'Existing median/MAD/quantization-floor masks were independently built from image intensity '
        '(not CAMs), with inclusive/primary/restrictive thresholds and brightness/uniform baselines. '
        'These are processed-intensity proxies, not independent validated cores, jets, lobes or hotspots. '
        'Annotation-based physics Jaccard is NOT AVAILABLE. Proxy IoU must not be relabeled as such.','',
        '| Primary proxy metric | Available | Mean | FR-I mean | FR-II mean |','|---|---:|---:|---:|---:|']
    for key in ['emission_attribution_mass','area_adjusted_enrichment','top10_emission_iou']:
        s=summary['metric_summaries']['proxy.primary.'+key]
        lines.append(f"| {key} | {s['overall']['available']} | {format_value(s['overall']['mean'])} | "
                     f"{format_value(s['by_class'][0]['mean'])} | {format_value(s['by_class'][1]['mean'])} |")
    lines+=['','## Deterministic representative figures','',
        'Lowest global saved Step20 index per ground-truth/correctness stratum, frozen before CAM '
        'generation. No selection by attractive heatmap, confidence or metric result.','']
    for group,index in protocol['representatives'].items():
        lines.append(f'- {group}: index {index}; representatives/index_{index}.png.')
    lines+=['','Figures show original fixed-scale intensity, normalized CAM and alpha .45 overlay, '
        'and are labeled exploratory. Core/jet/lobe interpretations require independent morphology '
        'annotations and imaging context. Intended language: radio emission structures produced by '
        'AGN activity, indirectly linked to supermassive black holes. This is not black-hole detection. '
        'Qualitative observations after figure inspection are recorded separately; no expert localization '
        'or causal interpretation is inferred from the numbers.','',
        '## Deferred/unavailable analyses','']
    lines.extend(f'- {name}: {reason}.' for name,reason in summary['unavailable_or_deferred'].items())
    lines+=['','Full 21-point/AUC and replacement sensitivities are compatible but deferred secondary '
        'resource scope (ten-fold deleted-input workload); parameter randomization would add 3480 '
        'gradient explanations. Neither was performed. This is not an assertion of mathematical '
        'incompatibility or completion. Four inherited convention gaps remain unresolved. ECE bins '
        'were not invented; corrected OOD diagnostics were not replaced with historical calibration.','',
        '## Runtime, integrity and limitations','',
        f"Process wall time {process['wall_seconds']:.3f}s ({process['wall_seconds']/60:.2f}min), "
        f"from {process['started_utc']} to {process['finished_utc']}; exit code 0. Per-image median "
        f"{np.median(durations):.3f}s, range {durations.min():.3f}–{durations.max():.3f}s. "
        f"Maximum process-lifetime peak working set {peak} bytes ({peak/1024**3:.3f}GiB); "
        f"maximum sampled working set {maximum_sampled} bytes. Family totals (exclusive seconds): {dict(family)}. "
        'Setup, aggregation and uninstrumented overhead are not included in these family sums. '
        '2GiB/900s cooperative checks passed; blocking operations cannot be interrupted immediately. '
        'Operational telemetry is separate from deterministic scientific records.','',
        'All 41 protected artifacts, existing snapshotted files, checkpoints, manifests, historical '
        'protocols, Step18–20 results and Git index/HEAD unchanged. No TRAIN/TEST image access, '
        'training, new classification evaluation, UI edit or staging/commit/push. Provenance reads '
        'include opaque serialized archive/split hashes, not TEST label/pixel interpretation.','',
        f"Completion snapshot checksum differences: {mismatch or 'none'}. "
        'The operational stdout log received the final completion line after the engine snapshot. '
        'Its prefix hash was verified against that snapshot; all deterministic scientific artifact '
        'checksums match. final_integrity.json records final post-process hashes without altering '
        'the original completion evidence. This is a log-publication ordering limitation, not a '
        'scientific record change.','',
        'One validation-selected model/seed, small error groups, validation reuse, unverified '
        'source independence/near duplicates, coarse positive-only CAM, deletion distribution shift, '
        'normalization/threshold/tie dependence and clipping constrain interpretation. No external '
        'or TEST generalization, physical correctness, model superiority or causality is established.','',
        'The essential corrected evaluation scope is complete; broader frozen-secondary coverage '
        'remains incomplete. Before research completion: researcher review of class/correctness '
        'differences, transparently report all limitations/deferred analyses and decide whether '
        'further sanity/sensitivity work is required. Any additional experiment, expert annotation '
        'exercise, TEST evaluation or UI integration requires separate authorization.','']
    with (D/'gradcam_evaluation_report.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    final=dict(status='PASS_SCIENTIFIC_ARTIFACTS_WITH_OPERATIONAL_LOG_LIMITATION',
        completed_images=116,train_or_test_images_decoded=0,protected_artifacts_unchanged=41,
        all_scientific_completion_hashes_match=True,completion_operational_log_mismatches=mismatch,
        no_additional_inference=True,peak_working_set_bytes=peak,maximum_sampled_working_set_bytes=maximum_sampled,
        per_image_seconds=dict(median=float(np.median(durations)),minimum=float(durations.min()),maximum=float(durations.max())),
        family_seconds=dict(family),negative_cam_probability_aopc_count=negative,
        final_output_sha256={p.relative_to(D).as_posix():sha(p) for p in D.rglob('*') if p.is_file()})
    with (D/'final_integrity.json').open('x',encoding='utf-8') as stream:
        json.dump(final,stream,indent=2)
    print(json.dumps(dict(images=116,statuses=summary['gradcam_status_counts'],runtime_seconds=process['wall_seconds'],
        primary_aopc={name:summary['metric_summaries'][name]['overall'] for name in selected},
        stability_counts=summary['stability_case_counts'],peak_bytes=peak,negative_aopc=negative,
        stdout_snapshot_note=bool(mismatch)),indent=2))


if __name__=='__main__':
    main()
