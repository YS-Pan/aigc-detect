"""Formula equivalence, coverage and safe weight/redirect contracts."""

import io
import json
import urllib.error
from unittest.mock import patch

import pytest
import torch
from transformers import RobertaConfig

from aigc_detector.assets import assets_installed
from aigc_detector.catalog import CATALOG, CAUSAL_ASSETS, EXTRA_ENGLISH_MODELS, ENGLISH_MODELS
from aigc_detector.engine import Detector, ModelUnavailable, select_models, window_ranges
from aigc_detector.modeling import LegacyRobertaPoolerClassifier
from aigc_detector.fusion import ENGLISH_POLICY, load_profile
from aigc_detector.statistical import distribution_components, inverse_coverage, raw_statistic
from scripts.download_models import model_assets
from scripts import network


def test_extra_models_cannot_silently_change_frozen_fusion():
    profile, _ = load_profile(ENGLISH_POLICY)
    assert set(profile['models']) == set(ENGLISH_MODELS)
    assert not set(EXTRA_ENGLISH_MODELS) & set(ENGLISH_MODELS)
    assert select_models(list(EXTRA_ENGLISH_MODELS), 'en') == list(EXTRA_ENGLISH_MODELS)
    assert all(CATALOG[model]['ai_label_index'] == 0 for model in EXTRA_ENGLISH_MODELS)
    with pytest.raises(ValueError, match='Unknown model'):
        select_models(list(CAUSAL_ASSETS), 'en')


def test_statistical_formulas_match_independent_tensor_expressions():
    reference = torch.tensor([[1., 2., -1.], [3., -2., 1.], [-1., 0., 2.], [2., 1., -1.]])
    scoring = torch.tensor([[2., -1., 1.], [1., 2., -2.], [0., 2., 1.], [-1., 3., 0.]])
    ids = [1, 2, 1, 0]
    values = distribution_components(reference, scoring, ids, block_tokens=2)
    probabilities = reference[:-1].softmax(-1)
    log_probs = scoring[:-1].log_softmax(-1)
    means = (probabilities*log_probs).sum(-1)
    variances = (probabilities*log_probs.square()).sum(-1)-means.square()
    observed = log_probs.gather(-1, torch.tensor(ids[1:])[:,None]).squeeze(-1)
    expected_fast = float((observed.sum()-means.sum())/variances.sum().sqrt())
    expected_binoculars = float(torch.nn.functional.cross_entropy(scoring[:-1],torch.tensor(ids[1:])) /
                               (-(reference.softmax(-1)*scoring.log_softmax(-1)).sum(-1).mean()))
    ones = torch.ones(3)
    assert raw_statistic('fast-detect-gpt-falcon',values,ones,torch.ones(4),ids,99) == pytest.approx(expected_fast,abs=1e-6)
    assert raw_statistic('binoculars-falcon',values,ones,torch.ones(4),ids,99) == pytest.approx(expected_binoculars,abs=1e-6)


def test_legacy_pooler_checkpoint_loads_all_keys_without_changing_cls_formula(tmp_path):
    config=RobertaConfig(vocab_size=16,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,
                         intermediate_size=16,max_position_embeddings=32,hidden_dropout_prob=0,
                         attention_probs_dropout_prob=0,num_labels=2)
    model=LegacyRobertaPoolerClassifier(config).eval()
    model.save_pretrained(tmp_path)
    restored,loading=LegacyRobertaPoolerClassifier.from_pretrained(tmp_path,local_files_only=True,
                                                                  weights_only=True,output_loading_info=True)
    assert not any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    assert 'roberta.pooler.dense.weight' in restored.state_dict()
    inputs=torch.tensor([[0,5,6,2]])
    with torch.inference_mode():
        tokens=restored.roberta(inputs).last_hidden_state
        assert torch.allclose(restored(inputs).logits,restored.classifier(tokens),atol=1e-7)


def test_statistical_overlap_covers_each_prediction_exactly_once():
    ranges = list(window_ranges(1300,512,64))
    predicted = [(start,end-1) for start,end in ranges]
    weights = inverse_coverage(predicted,1299)
    reconstructed = torch.zeros(1299,dtype=torch.float64)
    for start,end in predicted:
        reconstructed[start:end] += weights[start:end]
    assert torch.equal(reconstructed,torch.ones_like(reconstructed))
    with pytest.raises(ValueError,match='Uncovered'):
        inverse_coverage([(0,3),(4,7)],7)


def test_classifier_cache_releases_oldest_under_memory_pressure(tmp_path):
    detector=Detector(device='cpu',model_dir=tmp_path)
    detector._loaded={'oldest':object(),'recent':object()}
    with patch('torch.cuda.empty_cache'),patch('torch.cuda.mem_get_info',side_effect=[(1,100),(50,100)]):
        detector._ensure_cuda_room(40)
    assert list(detector._loaded)==['recent']
    detector._loaded.clear()
    with patch('torch.cuda.empty_cache'),patch('torch.cuda.mem_get_info',return_value=(1,100)), \
            pytest.raises(ModelUnavailable,match='Insufficient'):
        detector._ensure_cuda_room(40)


def test_sharded_assets_are_complete_and_exclude_remote_code(tmp_path):
    names=['config.json','tokenizer.json','model.safetensors.index.json',
           'model-00001-of-00002.safetensors','model-00002-of-00002.safetensors',
           'pytorch_model.bin.index.json','pytorch_model-00001-of-00002.bin',
           'modeling_falcon.py','other/model-00001-of-00002.safetensors']
    selected={name for name,_ in model_assets({'siblings':[{'rfilename':name} for name in names]})}
    assert selected==set(names[:5])
    files={}
    for name in selected:
        content=json.dumps({'weight_map':{'a':'model-00001-of-00002.safetensors','b':'model-00002-of-00002.safetensors'}}) if name.endswith('index.json') else 'dummy'
        (tmp_path/name).write_text(content,encoding='utf-8')
        files[name]={'bytes':(tmp_path/name).stat().st_size}
    entry={'installed':True,'files':files}
    assert assets_installed(tmp_path,entry)
    (tmp_path/'model-00002-of-00002.safetensors').unlink()
    assert not assets_installed(tmp_path,entry)


def test_origin_proxy_redirect_still_tries_cdn_direct():
    redirect=urllib.error.HTTPError('https://origin.test/file',302,'redirect',{'Location':'https://cdn.test/file'},io.BytesIO())
    response=io.BytesIO(b'weights')
    response.status=200
    with patch.object(network,'FAILED_DIRECT',{}),patch.object(network,'EVENTS',[]), \
            patch.object(network.DIRECT,'open',side_effect=[urllib.error.URLError('blocked'),response]) as direct, \
            patch.object(network.FALLBACK,'open',side_effect=redirect) as fallback:
        assert network.open_public('https://origin.test/file').read()==b'weights'
        assert direct.call_count==2 and fallback.call_count==1
        assert direct.call_args.args[0].full_url=='https://cdn.test/file'
        assert network.EVENTS[-1]['route']=='direct'


def test_parallel_ranges_resume_and_reject_wrong_content_range(tmp_path):
    data=bytes(range(101))
    partial=tmp_path/'weights.partial'
    partial.write_bytes(data[:31])
    def response(url,*,headers):
        first,last=map(int,headers['Range'][6:].split('-'))
        value=io.BytesIO(data[first:last+1])
        value.status=206
        value.headers={'Content-Range':f'bytes {first}-{last}/{len(data)}'}
        return value
    with patch.object(network,'open_public',side_effect=response):
        network._parallel_download('https://test/weights',partial,'pinned',len(data),3,part_size=17)
    assert partial.read_bytes()==data
    def wrong_response(*args,**kwargs):
        value=io.BytesIO(b'bad')
        value.status=200
        value.headers={}
        return value
    with patch.object(network,'open_public',side_effect=wrong_response),pytest.raises(RuntimeError,match='exact byte range'):
        network._parallel_download('https://test/bad',tmp_path/'bad.partial','pinned',3,2,part_size=17)
