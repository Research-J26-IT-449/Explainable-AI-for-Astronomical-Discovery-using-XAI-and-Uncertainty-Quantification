"""One approved corrected-XAI evaluation; existing mathematical cores unchanged."""
import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import yaml
from src.data import corrected_protocol as cp
from src.data.corrected_training_data import CorrectedArchiveReader
from src.data.preprocess import PreprocessConfig, build_transforms
from src.models.corrected_training import load_cohort, load_profile
from src.models.resnet_cnn import ResNetCNN
from src.utils.seed import set_global_seed
from src.xai import gradcam, faithfulness as faith, stability, morphology_audit as morphology, statistics as stats
from src.xai.execution_adapter import memory_snapshot

PROTOCOL = "Corrected-GradCAM-Validation-v1.0.0"
MANIFEST_HASH = "4fe53fae197ff3c757afb7b02164eea3b9ea8588c5a3a3bfcd14aba3c6afcf91"
CHECKPOINT_HASH = "d5841b91b8cdcdb0fb8a538b9a8c79955a7bf83b4a3dcb3bf8cc4f176f6f687c"
CHECKPOINT = ROOT / "outputs/models/mirabest_development_exactdedup_v1.0.0/corrected_seed42_01/models/resnet_cnn_best.pth"
STEP20 = ROOT / "analysis_reports/step20_corrected_validation_classification_attempt02"
UNAVAILABLE = {
    "annotation_physics_jaccard": "independent_validated_region_annotations_not_available",
    "secondary_stability_19x19": "native_grid_pooling_partial_support_ties_unspecified",
    "random_emission_baseline": "random_native_to_input_projection_unspecified",
    "sanity_brightness_map": "sanity_brightness_definition_unspecified",
    "sanity_random_map": "sanity_random_distribution_grid_seeds_replicates_projection_unspecified",
    "ece": "no_frozen_confidence_bin_definition",
    "corrected_ood_diagnostics": "historical_calibration_not_validated_for_corrected_checkpoint",
    "full_deletion_auc_and_replacement_sensitivities": "deferred_secondary_tenfold_deletion_workload",
    "parameter_randomization": "deferred_secondary_3480_additional_cam_calls",
    "confirmatory_tests": "corrected_development_analysis_exploratory_only",
}


def primitive(value):
    if is_dataclass(value):
        return primitive(asdict(value))
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    raw = cp.canonical_bytes(primitive(value))
    with Path(path).open("xb") as stream:
        stream.write(raw)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()


def git_state():
    return dict(branch=git("branch", "--show-current"), head=git("rev-parse", "HEAD"),
                index_sha256=sha(ROOT / ".git/index"), status=git("status", "--short"))


def configuration(rows, representatives):
    return dict(protocol=PROTOCOL, dataset_protocol=cp.PROTOCOL,
        manifest_reference=cp.OUTPUT_REFERENCE + "/manifest.json", manifest_sha256=MANIFEST_HASH,
        checkpoint_reference=CHECKPOINT.relative_to(ROOT).as_posix(), checkpoint_sha256=CHECKPOINT_HASH,
        validation_indices=[r.dataset_index for r in rows], supports={"0": 51, "1": 65},
        representatives=representatives, representative_rule="lowest index per ground-truth x correctness stratum, from Step20",
        target_layer=gradcam.TARGET_LAYER, target="original predicted-class pre-softmax logit; fixed across interventions",
        preprocessing=asdict(PreprocessConfig()), mathematical_parent="Phase3B-XAI-v1.0.0",
        inherited_tie_namespace=faith.PROTOCOL, primary_fractions=faith.FRACTIONS,
        exact_pixel_budgets=faith.BUDGETS, random_deletion_seeds=faith.RANDOM_SEEDS,
        replacements_executed=["border_median"], transformations=stability.protocol_transformations(),
        stability_top_fractions=stability.TOP_FRACTIONS, emission_variants=morphology.VARIANTS,
        unavailable_or_deferred=UNAVAILABLE, max_raw_batch=8, memory_limit_bytes=2*1024**3,
        cooperative_image_time_limit_seconds=900, execution_authority="Researcher Step21B explicit authorization")


def prepare():
    assert Path.cwd().resolve() == ROOT
    assert git("branch", "--show-current") == "feature/external-radio-input"
    assert not any((OUT/name).exists() for name in ["preflight.json", "complete.json", "failure.json", "records", "maps", "telemetry"])
    load_profile(ROOT)
    cohort = load_cohort(ROOT)
    assert cohort.manifest_sha256 == MANIFEST_HASH
    rows = tuple(r for r in cohort.rows if r.split == "validation")
    assert len(rows) == len({r.dataset_index for r in rows}) == 116
    assert Counter(r.label for r in rows) == {0:51, 1:65}
    assert not set(cp.QUARANTINE) & {r.dataset_index for r in rows}
    assert sha(CHECKPOINT) == CHECKPOINT_HASH
    assert sha(ROOT / cp.ARCHIVE_REFERENCE) == cp.FROZEN_PINS.archive_sha256
    saved = read(STEP20 / "completion_integrity.json")
    assert all(sha(STEP20/name) == h for name,h in saved["output_sha256"].items())
    with (STEP20 / "validation_predictions.csv").open(newline="") as stream:
        predictions = list(csv.DictReader(stream))
    assert [int(p["dataset_index"]) for p in predictions] == [r.dataset_index for r in rows]
    assert [int(p["ground_truth_label"]) for p in predictions] == [r.label for r in rows]
    representatives = {f"class{label}_{case}": min(int(p["dataset_index"]) for p in predictions
        if int(p["ground_truth_label"])==label and (p["correct"]=="True")==correct)
        for label in (0,1) for case,correct in [("correct",True),("incorrect",False)]}
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert state["epoch"] == 30
    set_global_seed(42)
    model = ResNetCNN().cpu()
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.requires_grad_(False)
    model.eval()
    assert isinstance(model.blocks[2].convolution[3], torch.nn.Conv2d)
    _, transform = build_transforms(PreprocessConfig())
    assert [type(t).__name__ for t in transform.transforms] == ["ToTensor","Normalize"]
    protected = read(STEP20 / "preflight_provenance.json")["protected"]
    assert len(protected)==41 and all(sha(ROOT/name)==h for name,h in protected.items())
    config = primitive(configuration(rows, representatives))
    if (OUT / "protocol.json").exists():
        assert read(OUT / "protocol.json") == config
    snapshot = memory_snapshot()
    assert type(snapshot.get("working_set_bytes")) is int and snapshot["working_set_bytes"] < 2*1024**3
    return rows, predictions, representatives, model, transform, protected, config


class Resources:
    def __init__(self):
        self.started=time.perf_counter()
        self.families={}
        self.peak=0
        self.sampled_peak=0
        self.check()

    def check(self):
        memory=memory_snapshot()
        working=memory.get("working_set_bytes")
        if type(working) is not int or working<=0:
            raise RuntimeError("Reliable memory measurement unavailable")
        self.peak=max(self.peak,memory.get("process_peak_bytes",working))
        self.sampled_peak=max(self.sampled_peak,working)
        if working>2*1024**3 or time.perf_counter()-self.started>900:
            raise RuntimeError("Cooperative image resource limit exceeded")

    @contextmanager
    def family(self,name):
        self.check()
        start=time.perf_counter()
        try:
            yield
        finally:
            self.families[name]=self.families.get(name,0)+time.perf_counter()-start
            self.check()

    def report(self):
        return dict(elapsed_seconds=time.perf_counter()-self.started, family_seconds=self.families,
                    process_lifetime_peak_bytes=self.peak, maximum_sampled_working_set_bytes=self.sampled_peak,
                    limits="2GiB working set, 900s/image; cooperative, no blocking-operation interruption")


def check_state(model, initial):
    assert all(not module.training for module in model.modules())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())


def figure(image, explanation, row, record):
    normalized=explanation.heatmap.numpy()
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    axes[0].imshow(image,cmap="gray",vmin=0,vmax=1)
    axes[0].set_title("Original radio intensity")
    axes[1].imshow(normalized,cmap="magma",vmin=0,vmax=1)
    axes[1].set_title("Normalized Grad-CAM")
    axes[2].imshow(image,cmap="gray",vmin=0,vmax=1)
    axes[2].imshow(normalized,cmap="magma",vmin=0,vmax=1,alpha=.45)
    axes[2].set_title("Overlay (alpha=0.45)")
    for axis in axes:
        axis.set_axis_off()
    fig.suptitle(f"Exploratory, not validated scientific explanations | index {row.dataset_index}\n"
                 f"True FR-{'I' if row.label==0 else 'II'}; predicted FR-{'I' if record['predicted_class']==0 else 'II'}; "
                 f"confidence {record['confidence']:.4f}; CAM {explanation.status}",fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "representatives" / f"index_{row.dataset_index}.png",dpi=160)
    plt.close(fig)


def scalar_records(records):
    output=[]
    for record in records:
        metrics={}
        f=record["faithfulness"]
        if f["status"]=="available":
            metrics[stats.CAM]=stats.observation(f["gradcam_aopc"]["probability"])
            metrics[stats.BRIGHTNESS]=stats.observation(f["brightness_aopc"]["probability"])
            metrics[stats.RANDOM]=stats.aggregate_repeats({int(k):v["probability"] for k,v in f["random_aopc"].items()},"random_deletion")
        else:
            for name in [stats.CAM,stats.BRIGHTNESS,stats.RANDOM]:
                metrics[name]=stats.observation(reason=f["reason"])
        for comparison in record["stability"]:
            spec=comparison["transformation"]
            if spec["kind"]!="noise":
                name=f"stability.{spec['kind']}.{spec['value']}"
                for j in comparison["jaccard"]:
                    metrics[f"{name}.top{j['fraction']}"]=stats.observation(j["similarity"],reason=None if j["similarity"] is not None else comparison["reason"])
        for sigma in stability.NOISE_SIGMAS:
            comparisons=[v for v in record["stability"] if v["transformation"]["kind"]=="noise" and v["transformation"]["value"]==sigma]
            for q in stability.TOP_FRACTIONS:
                repeats={v["transformation"]["seed"]:stats.observation(next(j["similarity"] for j in v["jaccard"] if j["fraction"]==q),reason=None if v["status"]=="valid" else v["reason"]) for v in comparisons}
                metrics[f"stability.noise.{sigma}.top{q}"]=stats.aggregate_repeats(repeats,"noise",sigma)
        for variant in record["emission_proxy"]:
            for name in ["emission_attribution_mass","area_adjusted_enrichment","top10_emission_iou"]:
                value=variant["gradcam"][name]
                metrics[f"proxy.{variant['proxy']['variant']}.{name}"]=stats.observation(value,reason=None if value is not None else ";".join(variant["gradcam"]["reasons"]) or "topk_unavailable")
        output.append(stats.image_record(record["dataset_index"],record["ground_truth_label"],record["predicted_class"],metrics))
    return stats.add_primary_contrasts(output)


def execute(prepared):
    rows,predictions,representatives,model,transform,protected,config=prepared
    started=datetime.now(timezone.utc).isoformat()
    tick=time.perf_counter()
    initial_git=git_state()
    existing=set(ROOT/name for name in protected)
    for base in ["src","scripts","tests","config","docs","demo","data/manifests","outputs/models","outputs/metrics","outputs/ood","outputs/xai",
                 "analysis_reports/step18_corrected_cnn_training","analysis_reports/step19_corrected_validation_analysis",
                 "analysis_reports/step20_corrected_validation_classification","analysis_reports/step20_corrected_validation_classification_attempt02",
                 "analysis_reports/step21_corrected_gradcam_evaluation"]:
        existing.update(p for p in (ROOT/base).rglob('*') if p.is_file() and "__pycache__" not in p.parts)
    baseline={p.relative_to(ROOT).as_posix():sha(p) for p in existing}
    modules=[gradcam,faith,stability,morphology,stats]
    implementation={Path(m.__file__).relative_to(ROOT).as_posix():sha(m.__file__) for m in modules}
    for path in [Path(__file__),OUT/"protocol.json",OUT/"corrected_xai_protocol.md"]:
        implementation[path.relative_to(ROOT).as_posix()]=sha(path)
    write(OUT/"preflight.json",dict(status="passed",protocol=config,started_utc=started,git=initial_git,
        baseline=baseline,protected=protected,implementation_sha256=implementation,
        command=[sys.executable,"-B",str(Path(__file__).relative_to(ROOT)),"--execute"],
        python=sys.version,torch=str(torch.__version__),numpy=np.__version__,cpu_threads=torch.get_num_threads(),
        model_state_check="weights and all buffers checked exactly after every image; gradients absent",
        planned_primary_deletion_calls=116*22*6,planned_gradcam_calls=116*31))
    for name in ["records","maps","telemetry","representatives"]:
        (OUT/name).mkdir(exist_ok=False)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    reader=CorrectedArchiveReader(ROOT/cp.ARCHIVE_REFERENCE,rows,archive_sha256=cp.FROZEN_PINS.archive_sha256,archive_md5=cp.FROZEN_PINS.archive_md5)
    records=[]
    current=None
    resource=None
    try:
        for row,saved in zip(rows,predictions):
            current=row.dataset_index
            resource=Resources()
            record=dict(dataset_index=current,corrected_split="validation",ground_truth_label=row.label)
            with resource.family("original_gradcam"):
                raw=reader.read_selected([current])
                try:
                    with Image.frombytes("L",(150,150),raw[current]) as pil:
                        image=np.asarray(pil,dtype=np.float32).copy()/255.0
                        tensor=transform(pil).unsqueeze(0)
                finally:
                    raw.clear()
                original=gradcam.generate_gradcam(model,tensor,target_class=int(saved["predicted_label"]))
                probability=float(original.logits.softmax(1)[0,original.predicted_class])
                observed=original.logits.softmax(1)[0].numpy()
                expected=np.array([float(saved["fri_probability"]),float(saved["frii_probability"])])
                record.update(predicted_class=original.predicted_class,confidence=probability,
                    classification_correct=original.predicted_class==row.label,target_class=original.target_class,
                    gradcam_status=original.status,logits=original.logits[0].tolist(),layer=original.layer_name,
                    step20_probability_max_difference=float(np.max(np.abs(observed-expected))),
                    heatmap_reference=f"maps/index_{current}.npz")
                write(OUT/"telemetry"/f"prediction_check_{current}.json",record)
                assert original.predicted_class==int(saved["predicted_label"])==original.target_class
                assert np.allclose(observed,expected,atol=1e-5,rtol=0),"Step20 prediction/probability mismatch"
            with resource.family("primary_aopc"):
                if original.status=="valid":
                    rankings=faith.prepare_rankings(image,original.raw_cam.numpy(),current,cam_status=original.status)
                    def score(normalized):
                        resource.check()
                        with torch.inference_mode():
                            values=model(normalized).detach().clone()
                        assert torch.isfinite(values).all()
                        resource.check()
                        return values
                    result=faith.evaluate_rankings(image,rankings,score,replacement="border_median")
                    assert result.gradcam.target_class==original.target_class
                    assert np.allclose(result.gradcam.original_logits,original.logits[0].numpy(),atol=1e-5,rtol=0)
                    record["faithfulness"]=primitive(dict(status="available",result=result,
                        gradcam_aopc=faith.aopc(result.gradcam),brightness_aopc=faith.aopc(result.brightness),
                        random_aopc={s:faith.aopc(c) for s,c in result.random},contrasts=result.contrasts))
                else:
                    record["faithfulness"]={"status":"unavailable","reason":"original_gradcam_degenerate"}
            with resource.family("stability"):
                comparisons=[]
                for spec in stability.protocol_transformations():
                    resource.check()
                    changed=stability.apply_transform(image,spec,current)
                    explanation=gradcam.generate_gradcam(model,faith.normalize_intensity(changed.image),original.target_class)
                    comparisons.append(stability.compare_explanations(original,explanation,changed,current))
                    resource.check()
                record["stability"]=primitive(comparisons)
            with resource.family("emission_proxy"):
                alignment=morphology.audit_emission_alignment(image,original.upsampled_heatmap,current,cam_status=original.status)
                record["emission_proxy"]=[]
                for variant in alignment.variants:
                    item=primitive(variant)
                    del item["proxy"]["emission_mask"]
                    record["emission_proxy"].append(item)
            check_state(model,initial)
            resource.check()
            assert all(sha(ROOT/p)==h for p,h in implementation.items())
            with (OUT/record["heatmap_reference"]).open("xb") as stream:
                np.savez_compressed(stream,raw_cam=original.raw_cam.numpy(),
                    upsampled_heatmap=original.upsampled_heatmap.numpy(),heatmap=original.heatmap.numpy())
            if current in representatives.values():
                with resource.family("representative_export"):
                    figure(image,original,row,record)
            resource.check()
            write(OUT/"records"/f"index_{current}.json",record)
            write(OUT/"telemetry"/f"index_{current}.json",resource.report())
            records.append(record)
            print(json.dumps(dict(completed=len(records),total=116,index=current,cam_status=original.status,
                confidence=probability,seconds=round(resource.report()["elapsed_seconds"],2))),flush=True)
        assert len(records)==len({r['dataset_index'] for r in records})==116
        scalar=scalar_records(records)
        names=sorted(set(k for record in scalar for k,_ in record.metrics))
        summaries={name:stats.summarize(scalar,name) for name in names}
        intervals={}
        for endpoint in stats.PRIMARY_FAMILY:
            interval=primitive(stats.bootstrap_paired(scalar,(endpoint,)))
            interval.pop("macro_replicates")
            interval.pop("overall_replicates")
            intervals[endpoint]=interval
        stability_counts={}
        for group in ["all","class0","class1","correct","incorrect"]:
            selected=[r for r in records if group=="all" or group==f"class{r['ground_truth_label']}" or (group=="correct")==r['classification_correct'] and group in ["correct","incorrect"]]
            values=[v for r in selected for v in r["stability"]]
            stability_counts[group]=dict(images=len(selected),comparisons=len(values),flips=sum(v['class_flip'] for v in values),
                unavailable=dict(Counter(v['reason'] for v in values if v['status']!='valid')))
        write(OUT/"quantitative_summary.json",dict(protocol=PROTOCOL,images=116,
            gradcam_status_counts=dict(Counter(r['gradcam_status'] for r in records)),
            metric_summaries=summaries,paired_aopc_bootstrap=intervals,
            stability_case_counts=stability_counts,unavailable_or_deferred=UNAVAILABLE,
            statistical_unit="image; random/noise repeats aggregated within image",
            scope="exploratory development; incomplete historical secondary coverage"))
        with (OUT/"explanation_metadata.csv").open("x",newline="",encoding="utf-8") as stream:
            columns=["dataset_index","corrected_split","ground_truth_label","predicted_class","confidence",
                     "classification_correct","target_class","gradcam_status","layer","heatmap_reference"]
            writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        assert all(sha(ROOT/p)==h for p,h in baseline.items())
        assert all(sha(ROOT/p)==h for p,h in protected.items())
        assert sha(CHECKPOINT)==CHECKPOINT_HASH and git_state()==initial_git
        check_state(model,initial)
        write(OUT/"complete.json",dict(status="complete_essential_scope_with_secondary_limitations",images=116,
            finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-tick,
            protected_artifacts_unchanged=41,git_unchanged=True,model_state_unchanged=True,
            checkpoint_sha256=CHECKPOINT_HASH,manifest_sha256=MANIFEST_HASH,implementation_sha256=implementation,
            decoded_indices=[r.dataset_index for r in rows],test_or_train_images_decoded=0,
            output_sha256={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file()}))
        print("COMPLETE: 116 corrected validation explanations; quantitative summary saved.",flush=True)
    except Exception:
        write(OUT/"failure.json",dict(status="failed",dataset_index=current,completed_records=len(records),
            traceback=traceback.format_exc(),resources=resource.report() if resource else None,
            instruction="Do not repeat or resume without separate approval; preserve all partial evidence"))
        raise


def synthetic_check():
    image=np.zeros((150,150),np.float32)
    image[60:90,65:85]=1
    cam=np.arange(361,dtype=np.float32).reshape(19,19)
    rankings=faith.prepare_rankings(image,cam,31)
    masks=faith.deletion_masks(rankings.gradcam)
    assert tuple(int(mask.sum()) for mask in masks)==faith.BUDGETS
    result=faith.evaluate_rankings(image,rankings,lambda x:torch.tensor([[float(x.mean()),0.]]))
    assert len(result.random)==20 and len(result.gradcam.deleted_logits)==6
    assert len(stability.protocol_transformations())==30
    spec=stability.TransformSpec("noise",.005,2000)
    assert np.array_equal(stability.apply_transform(image,spec,31).image,stability.apply_transform(image,spec,31).image)
    assert morphology.audit_emission_alignment(image,np.ones((150,150),np.float32),31).variants
    record=dict(dataset_index=31,ground_truth_label=0,predicted_class=0,faithfulness=primitive(dict(
        status="available",gradcam_aopc=faith.aopc(result.gradcam),brightness_aopc=faith.aopc(result.brightness),
        random_aopc={s:faith.aopc(c) for s,c in result.random})),stability=[],emission_proxy=[])
    assert scalar_records([record])[0].dataset_index==31
    print("Synthetic integration checks PASS; zero real images or model forward passes.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--synthetic-check",action="store_true")
    args=parser.parse_args()
    if args.synthetic_check:
        assert not args.execute
        synthetic_check()
        return
    prepared=prepare()
    if not (OUT/"protocol.json").exists():
        write(OUT/"protocol.json",prepared[-1])
    print("Preflight PASS: corrected checkpoint/cohort, fixed protocol, CPU memory and protected hashes.",flush=True)
    print("Representatives:",prepared[2],flush=True)
    if args.execute:
        execute(prepared)


if __name__=="__main__":
    main()
