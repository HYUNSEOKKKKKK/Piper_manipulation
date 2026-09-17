# Attribution and licenses

The original perception, native geometry, scripts and documentation contributed
by Piper_manipulation contributors are offered under Apache-2.0 (root `LICENSE`).
This does not relicense third-party software, model weights or the ROS package.

| Component | Origin | Applicable license / treatment |
|---|---|---|
| `ros2/piper_pnp` | [yejunjoo/piper_pnp](https://github.com/yejunjoo/piper_pnp), baseline `5c1a7d907e1b31128a79d5e422beb79fb85eada6`, originally maintained by yejun | BSD-3-Clause, as declared in upstream `package.xml` and `setup.py`; license text supplied in that directory. Includes substantial later cuboid, guarded-control, feed, recovery and placement changes. |
| MobileSAM / MobileSAMv2 | [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | Apache-2.0 at project root; component notices remain applicable. Downloaded separately at the pinned revision. |
| SAM components | [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything) | Apache-2.0; Meta copyright notices preserved by the MobileSAM checkout. |
| TinyViT | [microsoft/Cream](https://github.com/microsoft/Cream/tree/main/TinyViT) | MIT notices in the upstream source. |
| Ultralytics fork inside MobileSAMv2 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics), bundled version 8.0.120 | AGPL-3.0; file-level headers identify this license. The complete inference environment contains AGPL code. The root Apache license does not remove its requirements. |
| `third_party/patches/mobilesam-deployment.patch` | Changes to the above components | Apache-2.0 for MobileSAM build changes; AGPL-3.0 for the Ultralytics changes and detector-wrapper modifications. Modified-file notices are added by the patch. |
| AgileX ROS | [agilexrobotics/agx_arm_ros](https://github.com/agilexrobotics/agx_arm_ros) | Repository root MIT; `agx_arm_ctrl` declares Apache-2.0. Upstream notices are retained in the downloaded checkout. |
| AgileX URDF/meshes | [agilexrobotics/agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf), gitlink `f6642ce0d7872c686f29c99e9e10cd23d1d49313` | MIT; initialized recursively at the version pinned by the ROS repository. |
| AgileX Python SDK | [agilexrobotics/pyAgxArm](https://github.com/agilexrobotics/pyAgxArm) | LGPL-3.0-only as declared by upstream setup metadata. Downloaded separately. |
| ArUco ROS interfaces | [JMU-ROBOTICS-VIVA/ros2_aruco](https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco) | MIT; message interfaces are reused for compatibility, without requiring printed ArUco markers. |

Source pins are in `third_party/dependencies.json`; checkpoint sources, exact sizes
and SHA-256 values are in `third_party/weights.json`. No third-party checkpoint is
redistributed in this Git repository. Download location does not change its terms.
The v2 mirror is explicitly third-party; the official archive is also linked.
Review upstream terms before redistributing a combined application or offering a
hosted service, especially the AGPL inference components.

The baseline ROS package did not include a standalone LICENSE file in the local
checkout; its two package metadata files explicitly declare BSD-3-Clause. This
release preserves that declaration and attributes its origin rather than claiming
exclusive authorship of the controller or vendor framework.
