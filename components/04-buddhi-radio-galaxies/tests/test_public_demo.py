"""Results-only publication checks; no research samples or inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from demo import app, research_results
from scripts.build_public_demo_bundle import build, ALLOWLIST

ROOT=Path(__file__).resolve().parents[1]


def test_verified_saved_values_and_aggregate_figure():
    values=research_results.results()
    saved=json.loads((ROOT/'analysis_reports/step20_corrected_validation_classification_attempt02/metrics.json').read_bytes())
    xai=json.loads((ROOT/'analysis_reports/step21_corrected_gradcam_evaluation_attempt02/quantitative_summary.json').read_bytes())
    assert values['accuracy']==saved['accuracy'] and values['macro_f1']==saved['macro']['f1']
    assert values['confusion_matrix']==[[38,13],[2,63]]
    assert values['aopc30']['Grad-CAM']==xai['metric_summaries']['cam_probability_aopc30']['overall']['mean']
    assert values['explanations']==116 and not values['live_inference_available']
    assert (research_results.ASSETS/'confusion_matrix.png').read_bytes()==(ROOT/'analysis_reports/step20_corrected_validation_classification_attempt02/confusion_matrix.png').read_bytes()


def test_public_callback_never_imports_or_loads_inference(monkeypatch):
    monkeypatch.setenv('RADIOMORPH_PUBLIC_DEMO','1')
    result=app.predict_for_ui_with_gradcam(object())
    assert result[1].startswith('UNSUPPORTED') and result[3:6]==(None,None,None)
    assert 'Live prediction is disabled' in result[2]
    with pytest.raises(PermissionError,match='disabled'):
        app.load_calibrated_ood_detector()


def test_public_gallery_excludes_private_images_even_when_requested(monkeypatch):
    monkeypatch.setenv('RADIOMORPH_PUBLIC_DEMO','1')
    monkeypatch.setenv('RADIOMORPH_SHOW_LOCAL_EXAMPLES','1')
    assert research_results.local_examples()==[]


def test_private_gallery_references_existing_precomputed_figures_only(monkeypatch):
    monkeypatch.setenv('RADIOMORPH_PUBLIC_DEMO','0')
    monkeypatch.setenv('RADIOMORPH_SHOW_LOCAL_EXAMPLES','1')
    values=research_results.local_examples()
    assert len(values)==4
    assert [Path(p).name for p,_ in values]==['index_35.png','index_31.png','index_50.png','index_437.png']
    assert all(Path(p).is_file() for p,_ in values)


def test_asset_tampering_fails_closed(tmp_path,monkeypatch):
    source=research_results.ASSETS
    for name in ['research_results.json','provenance.json','confusion_matrix.png']:
        (tmp_path/name).write_bytes((source/name).read_bytes())
    (tmp_path/'research_results.json').write_text('{}')
    monkeypatch.setattr(research_results,'ASSETS',tmp_path)
    with pytest.raises(ValueError,match='checksum mismatch'):
        research_results.results()


def test_upload_controls_disabled_and_saved_results_visible(monkeypatch):
    monkeypatch.setenv('RADIOMORPH_PUBLIC_DEMO','1')
    monkeypatch.setenv('GRADIO_ANALYTICS_ENABLED','False')
    demo=app.create_demo()
    try:
        config=demo.get_config_file()
        components=config['components']
        for name in ['rm-upload','rm-mode','rm-analyze']:
            component=next(c for c in components if c['props'].get('elem_id')==name)
            assert component['props']['interactive'] is False
        text=' '.join(str(c['props'].get('value','')) for c in components)
        for value in ['87.068966%','0.864391','0.294510','0.053960','0.222254','redistribution']:
            assert value in text
        assert not any(c['type']=='gallery' for c in components)
        assert any('Saved corrected validation confusion matrix' in str(c['props'].get('label','')) for c in components)
    finally:
        demo.close()


def test_bundle_allowlist_no_private_assets_no_overwrite(tmp_path):
    destination=tmp_path/'bundle'
    ledger=build(destination)
    assert set(ledger)==set(ALLOWLIST.values())
    assert len(list(destination.rglob('*.png')))==1
    assert not any(p.suffix in {'.pth','.pt','.npz','.pkl','.jpg','.jpeg'} for p in destination.rglob('*'))
    assert not any(p.name in {'data','src','outputs','analysis_reports','.git','.env'} for p in destination.rglob('*'))
    for name,digest in ledger.items():
        assert hashlib.sha256((destination/name).read_bytes()).hexdigest()==digest
    with pytest.raises(FileExistsError):
        build(destination)
    for name in ['app.py','README.md','requirements.txt','Dockerfile']:
        text=(destination/name).read_text(encoding='utf-8')
        assert 'C:\\Users' not in text and 'hf_' not in text and 'sk-proj-' not in text
    text=(destination/'README.md').read_text(encoding='utf-8')
    assert 'sdk: docker' in text and 'app_port: 7860' in text


def test_standalone_bundle_constructs_without_research_or_torch(tmp_path):
    destination=tmp_path/'bundle'
    build(destination)
    code='''import sys,os
sys.path.insert(0,sys.argv[1])
sys.modules['torch']=None
os.environ['RADIOMORPH_PUBLIC_DEMO']='1'
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
from demo.app import create_demo,predict_for_ui_with_gradcam
d=create_demo()
assert len(predict_for_ui_with_gradcam(object()))==7
assert 'src.inference.corrected_demo' not in sys.modules
d.close()
print('STANDALONE_RESULTS_ONLY_PASS')
'''
    process=subprocess.run([sys.executable,'-I','-B','-c',code,str(destination)],cwd=destination,capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr
    assert 'STANDALONE_RESULTS_ONLY_PASS' in process.stdout


def test_public_entry_forces_safe_mode(monkeypatch):
    monkeypatch.setenv('RADIOMORPH_PUBLIC_DEMO','0')
    monkeypatch.setenv('RADIOMORPH_SHOW_LOCAL_EXAMPLES','1')
    spec=importlib.util.spec_from_file_location('public_entry_fixture',ROOT/'public_demo/app.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert research_results.public_mode() and research_results.local_examples()==[]
