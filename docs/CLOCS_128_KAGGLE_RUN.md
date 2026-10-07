# Kaggle execution cell

Attach `clocs_128_training_update.zip` as a new private code Dataset alongside
the existing inputs. Kaggle may unpack it automatically. Keep the existing
working repository and 128-frame examples. Paste this as a Python notebook cell.
This runs two sequential 12-epoch experiments. Do not restart the session until
the result archive has been downloaded and its SHA-256 verified on the Mac.

```python
from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import torch

assert torch.cuda.is_available(), 'Enable GPU before running'
repo = Path('/kaggle/working/multimodal-perception')
matches = list(Path('/kaggle/input').rglob('train_clocs_128.py'))
assert len(matches) == 1, f'Expected one attached update, found {matches}'
source = matches[0].parent.parent
for relative in ('fusion/clocs_sampling.py', 'scripts/train_clocs_128.py'):
    destination = repo / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / relative, destination)
data = Path('/kaggle/working/clocs_yolo26_train_128')
inputs = Path('/kaggle/input/datasets/abhishekjain18/clocs-train-export-128-tar-gz')
assert (data / 'clocs_yolo26_train_128_summary.json').is_file(), 'Restore the downloaded target package first'
assert (repo / 'scripts/train_clocs_pilot.py').is_file(), 'Restore the earlier project code first'
env = dict(os.environ, PYTHONPATH=str(repo), CUDA_VISIBLE_DEVICES='0', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
runs = []
for ratio in (0, 8):
    output = Path(f'/kaggle/working/clocs_128_ratio{ratio}_seed20260928')
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'Existing run: {output}. Preserve and inspect it before rerunning.')
    runs.append(output)
    command = [sys.executable, '-m', 'scripts.train_clocs_128',
        '--data-dir', str(data),
        '--batch-manifest', str(repo / 'results/kitti/clocs/training_exports/clocs_train_export_128_manifest.json'),
        '--train-ids', str(inputs / 'splits/train.txt'),
        '--val-ids', str(inputs / 'splits/val.txt'),
        '--output-dir', str(output), '--negative-ratio', str(ratio),
        '--epochs', '12', '--device', 'cuda']
    subprocess.run(command, cwd=repo, env=env, check=True)
archive = Path('/kaggle/working/clocs_128_training_comparison.tar.gz')
with tarfile.open(archive, 'w:gz') as bundle:
    for run in runs:
        bundle.add(run, arcname=run.name)
digest = hashlib.sha256()
with archive.open('rb') as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(block)
print('Download:', archive)
print('SHA-256:', digest.hexdigest())
```

If either run fails, preserve its output and share the error; do not delete runs
or manufacture a PASS report. CPU tests have exercised the trainer, but this
release has not been run here on real KITTI data or Kaggle CUDA.

