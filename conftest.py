"""Use checkout sources rather than an unrelated installed ROS overlay."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
for folder in ('perception', 'ros2/piper_pnp', 'scripts', 'tests/perception', 'ros2/piper_pnp/test'):
    sys.path.insert(0, str(ROOT / folder))
