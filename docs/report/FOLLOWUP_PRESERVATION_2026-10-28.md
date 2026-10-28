# Preserve the completed public follow-up evidence

**Status: utility prepared; the user's Drive copy has not been executed or verified in this session.** Git publication and local table checks do not establish a Drive backup.

The utility preserves every public file under the 28 October case-analysis tree (including its rain, metadata and night subfolders), the new supervisor synthesis tree, the original arm pilot tree, and the linked discussion/reproduction files. It includes the existing public PNG/PDF figures; it does not attempt to recover raw sensor archives or full prediction files held separately.

## CPU Colab procedure

1. Select a CPU runtime and mount Drive with `from google.colab import drive; drive.mount('/content/drive')`.
2. Take the **full 40-character commit SHA** from the published follow-up commit. Use that value below; do not use `main` as an unpinned substitute.
3. Run the following cell. It downloads only the pinned standard-library preservation utility, then verifies/downloads the selected public evidence. It stops if the output already exists.

```python
from pathlib import Path
import subprocess
import sys
import urllib.request

COMMIT = "PASTE_FULL_FOLLOWUP_COMMIT_SHA"
assert len(COMMIT) == 40 and all(c in "0123456789abcdef" for c in COMMIT)
assert Path('/content/drive/MyDrive').is_dir(), 'Mount Drive first'
folder = Path('/content/drive/MyDrive/Major_Project/nuscenes_full_validation/supervisor_followup_20261028')
folder.mkdir(parents=True, exist_ok=True)
url = f'https://raw.githubusercontent.com/abjain182002/multimodal-perception/{COMMIT}/scripts/preserve_supervisor_followup.py'
script = Path('/content/preserve_supervisor_followup.py')
with urllib.request.urlopen(url, timeout=120) as response:
    script.write_bytes(response.read())
subprocess.run([sys.executable, str(script), '--commit', COMMIT, '--destination', str(folder)], check=True)
```

The expected completion marker is `PUBLIC_FOLLOWUP_ARCHIVE_AND_DESTINATION_COPY_VERIFIED`. The utility checks each downloaded file's size and Git blob SHA, checks the local ZIP, copies it to the destination, then checks the copied ZIP's SHA-256 and every payload member. It saves a receipt alongside the ZIP and does not overwrite an earlier archive or receipt. Retain the receipt as the preservation record.

This creates a new complete public-evidence snapshot, which can replace the need to copy several separate derived bundles. It does not certify the state of old Drive bundles, raw datasets or checkpoints, rerun numerical evaluation, or establish physical robot performance. No claim of Drive completion should be made before the receipt is available.
