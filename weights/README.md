These files are downloaded, never committed. Run `python3 scripts/fetch_weights.py`
from the repository root (188.3 MiB total). SHA-256 verification is mandatory.

MobileSAM's encoder comes from the official repository. The two v2 checkpoints
use an explicitly identified, revision-pinned third-party mirror. Alternatively,
extract the [official archive](https://drive.google.com/file/d/1dE-YAG-1mFCBmao2rHDp0n-PP4eH7SjE/view)
and run `python3 scripts/fetch_weights.py --from-dir /path/to/extracted/weights`.
Place `mobile_sam.pt` from the official MobileSAM repository in that folder too.

`python3 scripts/fetch_weights.py --list` lists exact URLs and sizes;
`--verify` validates an existing installation without network access.
See `third_party/weights.json` and `NOTICE.md` for provenance and licensing.
The YOLO checkpoint contains Python pickle objects: do not substitute untrusted files.
