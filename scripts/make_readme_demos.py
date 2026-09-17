#!/usr/bin/env python3
"""Export small, silent README previews from the original experiment recordings.

This utility only processes files; it never starts ROS or controls hardware.
Requires ffmpeg on PATH, or --ffmpeg /path/to/ffmpeg.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CLIPS = (
    ('IMG_8200.MP4', 'separated_boxes.gif', 2.0, 47.0, 'Separated boxes: one complete recorded feed cycle'),
    ('IMG_8201.MP4', 'stacked_boxes.gif', 1.0, 45.0, 'Stacked boxes: first recorded feed cycle'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'assets')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source_name, output_name, start, duration, description in CLIPS:
        source, output = args.source_dir / source_name, args.output_dir / output_name
        if not source.is_file():
            raise SystemExit(f'Missing recording: {source}')
        # Preserve the full selected interval; 4x playback and spatial downsampling only.
        filters = ('setpts=(PTS-STARTPTS)/4,fps=8,scale=400:-1:flags=lanczos,split[a][b];'
                   '[a]palettegen=max_colors=96:stats_mode=diff[p];'
                   '[b][p]paletteuse=dither=bayer:bayer_scale=3')
        subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                        '-ss', str(start), '-t', str(duration), '-i', str(source),
                        '-an', '-filter_complex', filters, '-loop', '0',
                        '-map_metadata', '-1', '-threads', '2', str(output)], check=True)
        if output.stat().st_size > 5 * 1024 * 1024:
            raise SystemExit(f'{output}: exceeds the repository media budget')
        manifest.append(dict(file=output_name, source_recording=source_name,
                             source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             start_s=start, duration_s=duration, playback_speed=4,
                             fps=8, width_px=400, description=description,
                             controller_mode='Historical nine-stage feed; operator approval per cycle',
                             sha256=hashlib.sha256(output.read_bytes()).hexdigest(), bytes=output.stat().st_size))
        print(f'{output.name}: {output.stat().st_size / 1048576:.2f} MiB')
    (args.output_dir / 'demos.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
