"""Recheck cached batch outputs against individual offline local inference."""
import gc
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import english_specialized_fusion as sf
from scripts.calibrate_english_v2 import make_records


def main():
    import torch
    from aigc_detector.engine import Detector
    target = sf.EVIDENCE / "real-inference-validation.json"
    if target.exists():
        raise FileExistsError("Preserve the completed validation")
    sf.verify_freeze()
    samples = []
    for split, name in [(('fit', 'calibration'), 'development'), (('test',), 'repeat')]:
        records, passages, _ = make_records(1600, 400, split, ('human_document', 'mixed_document', 'ai_paragraph'))
        chosen = [next(r for r in records if r['kind'] == 'human_document'),
                  next(r for r in records if r['kind'] == 'mixed_document' and r['ai_chars'])]
        samples.extend((name, r, passages[r['sha256']]) for r in chosen)
    def blocked(*args, **kwargs):
        raise AssertionError("Inference attempted a network connection")
    detector, checks = Detector(device='cuda', batch_size=8), []
    started = time.perf_counter()
    with patch.object(socket.socket, 'connect', blocked), patch.object(socket, 'create_connection', blocked):
        for mid in sf.NEW:
            for name, row, text in samples:
                cached = sf.read(sf.REPORTS / name / f'{mid}-cache.json')['items'][row['sha256']]
                actual = detector.detect(text, [mid], language='en')['results'][0]
                assert actual['revision'] == cached['revision']
                differences = {'ai_score': abs(actual['ai_score']-cached['ai_score'])}
                for label, score in cached.get('class_scores', {}).items():
                    differences[label] = abs(actual['class_scores'][label]-score)
                assert max(differences.values()) < 2e-5, (mid, row['id'], differences)
                checks.append({'model_id': mid, 'input_id': row['id'], 'sha256': row['sha256'],
                               'revision': actual['revision'], 'max_abs_difference': max(differences.values())})
            detector._loaded.clear()
            gc.collect()
            torch.cuda.empty_cache()
    sf.write(target, {'status': 'passed', 'network_blocked': True, 'checks': checks,
                      'max_abs_difference': max(r['max_abs_difference'] for r in checks),
                      'gpu': torch.cuda.get_device_name(0), 'seconds': time.perf_counter()-started,
                      'contains_source_text': False})
    print(json.dumps({'passed': len(checks), 'max_abs_difference': max(r['max_abs_difference'] for r in checks)}))


if __name__ == '__main__':
    main()
