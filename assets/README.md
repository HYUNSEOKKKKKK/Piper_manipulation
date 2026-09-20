# README media

All experiment visuals here come from this project's own recordings or analysis.
The GMR reference inspired the README layout; its images, videos and results are
not used as PiPER experiment evidence.

| Asset | Description | Provenance / limits |
|---|---|---|
| `pipeline.svg` | Original vector schematic of the released pipeline | Illustrative camera, cuboid and arm symbols; not a measured scene or robot model |
| `separated_boxes.gif` | One recorded transfer, separated-box scene | `IMG_8200.MP4`, source interval 2–49 s, 4× playback |
| `stacked_boxes.gif` | First recorded transfer, stacked-box scene | `IMG_8201.MP4`, source interval 1–46 s, 4× playback |
| `feed_auto_run.gif` | One continuous `feed-auto` run, four transfers | `Final_manipulationtest.mp4`, full 88.5 s recording, 4× playback |
| `pose_comparison.png` | PnP and RGB-D refined overlays on two saved frames | Existing archive analysis, `frame_00` and `frame_04`, historical 77 × 35 × 30 mm box; no external ground truth |

`separated_boxes.gif` and `stacked_boxes.gif` were recorded with the historical
nine-stage feed sequence and an operator approving each cycle; they do not
demonstrate the current `feed-auto` runtime. `feed_auto_run.gif` is one continuous
`feed-auto` run started by a single approval, with each box fed onto the table by
hand between cycles. Each preview preserves its selected source interval at uniform 4×
speed; it is resized to 400 pixels wide at 8 fps, looped, and contains no audio.
The stacked preview ends after the first transfer; it does not show the later
manual removal of a dark plate, or claim complete autonomous pile clearing.

The complete original recordings are not stored in Git. Exact intervals, source
hashes, output hashes and encoding settings are recorded in [`demos.json`](demos.json).
To regenerate these previews from the original recordings:

```bash
python3 scripts/make_readme_demos.py --source-dir /path/to/recordings
```

`ffmpeg` must be on PATH, or pass `--ffmpeg /path/to/ffmpeg`. Run this command from
the repository root. It processes local files and sends no robot commands.
Project-produced media in this directory are distributed under the repository's
Apache-2.0 license. Component software licenses are described in [`NOTICE.md`](../NOTICE.md).
