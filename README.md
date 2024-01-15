# Northern_Hemisphere_Temperature

### 조원

- (조장)편하늘, 김범찬, 임지혜, 전정현, 한상준

### Environment Setting
- 실험 환경
  - Linux
  - Python 3.10

모든 패키지 및 라이브러리를 설치하려면, 터미널 창에서 `pip install -r requreiment.txt` 을 실행하세요

## 데이터셋 출처
### (https://www.kaggle.com/datasets/antoniomartin/northern-hemisphere-monthly-temperature-1880-2022)

## 모델설명 ppt
### (https://www.canva.com/design/DAF5cdjRJrE/bxy7mnaTv8QnUV3CryD4-A/edit)

### Dataset
- `data` 폴더 내에 전처리가 완료된 파일이 존재합니다(final.csv)
- 전체 데이터셋으로 학습을 원한다면, '데이터셋 출처'를 확인해주세요


### Start
```bash
# if rnn_model
# python main_lstm.py

# else
# python main.py
bash start.sh
```

### schedule
- 2024-01-02 데이터 선정 및 EDA & Arima 모델 적용

- 2024-01-03 transformer(PatchTST) 모델 적용

- 2024-01-04 LSTM 모델 적용

- 2024-01-05 LSTM 모델, 뉴럴네트워크, multichannel 모델 적용

- 2024-01-08 뉴럴네트워크, multichannel 모델 적용 및 모듈화 작업 진행

- 2024-01-09 모듈화 작업 완료 및 모델 테스트

- 2024-01-10 모듈화 최종본 업로드 및 발표 자료 준비

## git 관리 관련 참고할만한 사이트
- [A visual git reference](https://marklodato.github.io/visual-git-guide/index-ko.html)
- [간단하게 정리한 git 사용법](https://gin-girin-grim.tistory.com/10)

           