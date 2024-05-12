# VR 데이터 분석 프로그램

이 프로그램은 VR 환경에서 수집된 데이터를 분석하고 시각화하는 기능을 제공합니다. 프로그램은 다음과 같은 세 가지 파일로 구성되어 있습니다:

1. `test_filter_multifile_individual_mean.py`
2. `test_filter_multifile_spss.py`
3. `test_filter_multifile_total_mean.py`

## 주요 기능

### 1. test_filter_multifile_individual_mean.py

- 칼만 필터를 사용한 위치 및 방향 데이터 필터링
- 속도 계산
- 각 파일에 대한 통계 정보 계산 (최대값, 최소값, 평균, 표준편차, 변동계수, 적분값)
- 상관관계 계산
- 평균 데이터를 사용한 다양한 그래프 생성
  - 쿼터니언 시계열
  - 속도
  - 3D 위치 추적
  - 속도 크기
  - 가속도
  - 히스토그램
  - 산점도 매트릭스
- 처리된 데이터 및 통계 정보를 CSV 파일로 저장
- 생성된 그래프를 이미지 파일로 저장

### 2. test_filter_multifile_spss.py

- 개별 파일 분석 기능 (`test_filter_multifile_individual_mean.py`와 동일)
- 계산된 통계 정보를 데이터프레임으로 변환
- SPSS 분석을 위한 데이터 형식으로 데이터프레임 재구성
- 결측값 처리
- SPSS 분석용 데이터를 CSV 파일로 저장

### 3. test_filter_multifile_total_mean.py

- 개별 파일 분석 기능 (`test_filter_multifile_individual_mean.py`와 동일)
- 각 변수의 통계 정보에 대한 전체 평균 및 표준편차 계산
- 계산된 전체 평균 및 표준편차를 CSV 파일로 저장
- 전체 평균 데이터를 사용하여 다양한 그래프 생성
  - 쿼터니언 시계열
  - 속도
  - 3D 위치 추적
  - 속도 크기
  - 가속도
  - 히스토그램
  - 산점도 매트릭스

## 사용 방법

1. 프로그램 실행 시 파일 선택 창이 나타납니다. 분석할 입력 파일을 선택하세요.
2. 출력 디렉토리 선택 창이 나타납니다. 결과 파일이 저장될 디렉토리를 선택하세요.
3. 선택한 파일과 디렉토리가 유효한 경우, 프로그램이 데이터 처리 및 분석을 시작합니다.
4. 분석이 완료되면 선택한 출력 디렉토리에 처리된 데이터, 통계 정보, 그래프 이미지 파일이 저장됩니다.
5. 프로그램 실행 시간이 콘솔에 출력됩니다.

## 요구 사항

- Python 3.x
- NumPy
- Matplotlib
- Pandas
- Seaborn
- PyKalman

