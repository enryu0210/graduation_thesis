"""pytest 공통 설정 — src 하위 모듈을 import 할 수 있도록 경로를 추가한다.

이 프로젝트는 스크립트 실행(python src/...) 중심이라 별도 패키지 설치를 하지 않는다.
테스트에서 모듈을 import 하기 위해 src 와 src/imaging 를 sys.path 에 넣어준다.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for extra in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "src" / "imaging",
    PROJECT_ROOT / "src" / "models",
    PROJECT_ROOT / "src" / "eval",
    PROJECT_ROOT / "src" / "attacks",
    PROJECT_ROOT / "src" / "defense",
):
    path_str = str(extra)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
