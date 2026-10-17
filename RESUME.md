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
python -m pytest -q
```

로컬 인식 환경, 고정된 모델·업체 소스, native 기하 모듈과 robot SDK를 준비했습니다.
CPU 테스트는 128개 통과, ROS 전용 1개 skip이며 실제 detector/SAM CPU 추론도
38개 마스크를 생성했습니다. 두 Python 환경의 `pip check`도 통과했습니다.
이 PC의 ROS 2 시스템 패키지는 아직 설치되지 않았으며, sudo 비밀번호 입력이 필요해
ROS colcon 빌드와 ROS 메시지 의존 테스트는 보류 상태입니다. 시스템 설치 방법은
[INSTALL.md](docs/INSTALL.md), 이 컴퓨터의 이어서 실행할 명령은 `.local/README.md`에 있습니다.

이 PC의 로컬 터미널에서는 다음 명령이 관리자 인증 후 ROS 설치, 빌드, mocked 테스트를
이어서 수행합니다. 로봇이나 카메라는 시작하지 않습니다.

```bash
bash ~/Piper_manipulation/.local/finish_ros_setup.sh
```

CPU 예제/테스트와 모델 로딩 확인은 실제 로봇 구동 또는 성능 검증과 구분합니다.
공개 데모 원본 영상 3개는 확인한 로컬 자료 위치에서 찾지 못했으므로, 다른 PC에 있다면
출처·해시와 함께 별도로 보존해야 합니다. 촬영 장착/TCP/작업영역은 실제 장비에 맞춰
[ROBOT.md](docs/ROBOT.md)를 확인하세요.
