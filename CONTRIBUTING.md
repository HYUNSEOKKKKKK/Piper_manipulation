# Contributing

Keep perception geometry independent of ROS where possible. Add tests for
changed behavior, run the CPU suite and native/NumPy parity tests, and report
whether ROS mocked checks or physical experiments were performed. Do not claim
hardware validation from a mock.

Do not commit checkpoints, credentials, personal paths, recordings, state files,
colcon output or third-party binary distributions. Keep a dependency's exact
revision, license and patch source documented when changing it. Run
`python3 scripts/audit_release.py` before proposing a release.

Motion changes must preserve fresh feedback, explicit start, complete path
validation, collision checks, command gating and stop behavior. Include planning
and observation timing separately when reporting a performance improvement.
