# CV13 - 캠을 통한 가상 마우스

## update

<details>
<summary>200517 - 간단한 프로토타입 추가</summary>

* 간단한 프로토타입 추가
  * dependency: 
  
    `pip3 install opencv-python pyautogui mediapipe`
  * 실행: 
    
    `python3 prototype/test.py`

</details>

<details>
<summary>200601 - 프로토타입: 손동작 데이터 샘플링 & 손동작 분류 모델 추가</summary>

* 손동작 데이터 샘플링 코드
  * opencv, mediapipe를 이용해 손동작의 키포인트 데이터 샘플링
  * 손동작 분류 모델 학습 데이터로 이용
* 손동작 분류 모델 추가
  * 간단한 신경망을 통해 손동작 인식
  * 클릭, 우클릭 등의 마우스 동작을 손동작으로 표현하여 구현할 예정
</details>