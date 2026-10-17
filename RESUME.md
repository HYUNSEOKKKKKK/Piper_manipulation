# Piper 작업 재개 안내

정리 기준일: **2026-10-18**. 이 저장소를 인식과 PiPER 제어의 기준 프로젝트로 사용합니다.

## 코드와 환경

- `perception/cuboid_rgbd.py`: 알려진 박스 치수·RGB-D·마스크로 물체 중심 SE(3) 추정.
- `perception/ros_pose_bridge.py`: YOLO/MobileSAMv2 인식, ROS 영상 입력, 추정 결과 전달.
- `ros2/piper_pnp/`: 좌표 변환·관측 승인·MoveIt 계획·팔/그리퍼 제어와 launch 설정.
- 인식 환경: `conda activate piper_manipulation`.
- 제어용 Python: `.venv-robot/`; 시스템 ROS 2 Humble 설치와 `.workspace` 빌드가 필요합니다.
- 모델: `weights/`, 업체 소스: `third_party/MobileSAM/`와 `.workspace/`.
- 기존 환경 `defm`과 DeFM 소스는 다른 연구용으로 그대로 유지했습니다.

## 보존 자료

이전 실험 Git 저장소와 촬영 자료는 `.local/archive/manipulator/`, 전달용 Git bundle은
`.local/archive/manipulator_transfer.bundle`에 보존했습니다. 이전 실험 폴더 파일
901개의 이동 전후 SHA-256이 일치했습니다. 활성 가중치 3개도 공개 manifest와 일치합니다.

예전 `Perception/manipulator`와 `locomotion/third_party/piper_manipulation` 경로는
호환 링크입니다. 실제 코드·자료는 이 프로젝트 아래에 있습니다. `.local/README.md`에
실제 경로, 환경 내보내기와 검증 기록을 남깁니다. Git은 이 보존 자료를 백업하지 않습니다.

## 실행과 남은 단계

```bash
conda activate piper_manipulation
cd ~/Piper_manipulation
python examples/estimate_pose.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

이 PC의 시스템 ROS 2 Humble 설치와 프로젝트 ROS 6개 패키지 빌드를 완료했습니다.
제어 환경의 setuptools는 colcon 호환 버전 79.0.1이며 두 환경의 `pip check`가 통과했습니다.
ROS가 연결된 기본 테스트 129개, 전체 mocked ROS/인식 테스트 393개가 통과했습니다.
실제 인식 노드도 RTX 3070에서 CUDA/FP16·native 기하·모델 2개로 초기화했습니다.

가상 하드웨어에서 MoveIt 경로 계획, 팔 궤적 실행, Cartesian 계획과 그리퍼 액션이
성공했습니다. 보관된 RGB-D 12개 프레임도 실제 GPU 인식·SE(3) 코드로 재생했으며,
2개는 타겟을 출력하고 10개는 적합도 조건으로 거부했습니다. 오프라인 재생은 TF와
프레임 시간 제한을 해제한 카메라 좌표 진단이며 실기 정확도 측정은 아닙니다.

이 컴퓨터에서 인식 노드 초기화와 mocked 테스트를 다시 확인하려면:

```bash
bash .local/validate_ros_runtime.sh
```

실제 장비는 아직 연결하지 않았습니다. 다음 실기 세션은
[ROBOT.md](docs/ROBOT.md)의 카메라 USB 권한·CAN 1 Mbit/s·PC 제어 모드·영상/TF 확인을
거쳐 기존 3개 터미널로 시작합니다. 로컬 설치·검증 기록은 `.local/README.md`에 있습니다.

하드웨어 없는 소프트웨어 검증과 실제 로봇 구동·성능 검증은 구분합니다.
공개 데모 원본 영상 3개는 확인한 로컬 자료 위치에서 찾지 못했으므로, 다른 PC에 있다면
출처·해시와 함께 별도로 보존해야 합니다. 촬영 장착/TCP/작업영역은 실제 장비에 맞춰
[ROBOT.md](docs/ROBOT.md)를 확인하세요.
