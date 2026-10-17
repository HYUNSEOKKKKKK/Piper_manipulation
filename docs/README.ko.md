# 한국어 실행 안내

PiPER + 손목 D435로 알려진 크기의 박스를 인식하고 옮기는 연구 코드입니다.
YOLO 박스 검출 → MobileSAMv2 마스크 → RGB-D cuboid SE(3) 추정 → MoveIt 계획 →
PiPER 제어까지 포함합니다. 박스 전용 pose 네트워크를 추가 학습하지 않습니다.

처음 받았다면 [설치 문서](INSTALL.md)의 순서로 설치하세요. 로봇 없이 먼저
`examples/estimate_pose.py`로 추정을 확인할 수 있습니다. 가중치와 업체 코드는
Git에 넣지 않고 고정된 버전으로 별도 받습니다. `fetch_weights.py`가 다운로드한
파일의 크기와 SHA-256을 검사합니다. YOLO 포함 구성의 AGPL 조건은
[NOTICE.md](../NOTICE.md)에 따로 적었습니다.

기준 폴더는 `~/Piper_manipulation`, 인식용 Conda 환경 이름은
`piper_manipulation`입니다. PiPER 제어용 Python은 프로젝트 안의 `.venv-robot`,
모델은 `weights/`, ROS 소스와 빌드 결과는 `.workspace/`에서 관리합니다.
시스템 ROS 2 Humble 설치도 필요합니다. [폴더와 보존 자료](WORKSPACE.md),
[작업 재개 안내](../RESUME.md)를 참고하세요.

새 PC에서는 **clone → 시스템 ROS 설치 → 고정 소스 받기 → 제어용 Python 준비와 빌드
→ Conda 인식 환경·모델 준비 → 검증 → 장비 보정과 CAN 연결 → 실행** 순서입니다.
설치·빌드 명령은 [INSTALL.md](INSTALL.md), 첫 연결과 실행은 [ROBOT.md](ROBOT.md)에
있습니다. ROS는 `/opt/ros/humble`, 프로젝트 빌드 결과는 `.workspace/`에 놓입니다.
제어용 `.venv-robot`은 실행 스크립트가 선택하므로 별도로 활성화하지 않아도 됩니다.
ROS까지 Conda에 넣는 RoboStack 방식은 현재 저장소에서 검증한 설치 경로가 아닙니다.

전체 알고리즘에는 PiPER 팔·평행 그리퍼 외에도 손목의 RealSense D435와
SocketCAN 연결, 기본 실시간 인식 설정에 맞는 CUDA GPU 환경이 필요합니다.
카메라 장착·TCP·박스 치수·작업영역은 제3자의 실제 장비에 맞춰 보정해야 합니다.

실험 PC에서 쓰던 폴더와 독립적인 구조입니다. 기존 `~/run_robot.sh` 등의
개인용 바로가기를 덮어쓰지 않습니다. 이 저장소를 사용할 때는 저장소 루트에서:

```bash
# 터미널 1
bash scripts/run_robot.sh feed-auto
# 터미널 2
conda activate piper_manipulation
bash scripts/run_bridge.sh feed-auto
# 터미널 3
bash scripts/run_console.sh
```

카메라 장착 변환, TCP, 작업 영역, 그리퍼 오프셋을 자신의 장비에 맞춰야 합니다.
초기 설정의 두 박스는 78×35×30 mm, 78×53×30 mm이고, goal TCP는
(410, -180, 150) mm입니다. `feed-auto`는 c 한 번으로 최대 10개 연속 이송하며,
박스 사이 홈 복귀 없이 카메라 자세로 돌아와 다음 박스를 관측합니다.
놓기는 고정 XYZ로 이동해 공중에서 열고, 놓기 하강은 없습니다.

**c는 Enter 없이 누릅니다. Enter와 Space는 정지 키입니다.** 관측을 기다리는
중에도 다음 물체가 인식되면 움직일 수 있으므로, 박스를 추가하려면 먼저
정지하세요. h → y는 그리퍼와 goal을 비우고 손을 뺀 뒤 확인하는 홈/횟수 초기화입니다.
복구, CAN 오류, 티칭 모드, 부분 Cartesian 경로 오류는 [로봇 안내](ROBOT.md)에
정리했습니다.

터미널 2의 처리 Hz는 영상 처리 속도입니다. 로봇 한 사이클 속도와는 다릅니다.
추정 결과를 승인할 때 현재 설정은 서로 다른 유효 관측 4개, 3초 창,
위치 편차 20 mm와 회전 편차 15° 조건을 사용합니다.
