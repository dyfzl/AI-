# AI 전공특화교육 프로그램
**저자 스타일 기반 텍스트 생성 및 문체에 따른 감성 분석 차이**

<br/>

## 📖 프로젝트 소개
이 프로젝트는 특정 작가의 문체를 학습한 텍스트 생성 모델(GPT-2)을 구축하고, 생성된 텍스트의 감성을 분석하여 문체와 감성 표현 간의 관계를 분석하는 것을 목표로 한다. 


<br/>

* 특정 작가의 데이터를 학습하여 문체를 반영한 텍스트 생성 모델 구축
  
* 생성된 텍스트의 BLEU 점수 및 감성 분포 분석
  
* 문체 보존 여부가 감성 분석 성능에 미치는 영향 탐구
<br/>

----
### 🚀 코드 수행 과정
1. 데이터 로드 및 전처리
  * 데이터 출처: 데이콘 - 소설 문장 뭉치 데이터 이용 
  * 문장 언어: 영어 


* 전처리:
  - 특수문자와 불필요한 공백 제거
  - 쉼표, 마침표 등 문체 보존에 중요한 구두점 유지
  - 감성 라벨링: TextBlob으로 Positive, Neutral, Negative로 분류하고 숫자로 매핑

  #####    문체 보존 전처리 함수
  ```
  def clean_text_preserve_style(text):
      import re
      text = re.sub(r"[^\w\s.,!?']", "", text)  # 특수문자 유지
      text = re.sub(r"\s+", " ", text).strip()  # 공백 제거
      return text
  ```

  
2. 문체 학습 모델(GPT-2 Fine-Tuning)
모델 구성:
  - Hugging Face의 GPT-2 모델을 사용
  - 특정 작가(author=3)의 데이터를 학습
  - 학습 데이터(90%), 평가 데이터(10%)로 분할 후 Fine-tuning

    
  - 모델 학습:
  - 학습 손실 감소: 마지막 Epoch 기준 0.328
  - BLEU 점수 - 모델 성능 평가

    
  - 텍스트 생성:
  - 프롬프트 [We, The day was bright, she] 입력 시 텍스트 생성
  - We have no doubt about that, but I am sure you may very well be quite right.


    ##### GPT-2 모델 텍스트 생성 파이프라인
    ```
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    generated_text = text_generator("she", max_length=100, num_return_sequences=1)
    print("Generated Text:", generated_text[0]["generated_text"])
    ```


3. 감성 분석 모델(DistilBERT)
모델 구성:
  * TextBlob 감성 분석을 보완하기 위해 DistilBERT 기반 감성 분석 모델 구축
  * 학습 데이터(80%), 검증 데이터(10%), 테스트 데이터(10%)로 분할
  * 모델 학습 및 평가:
  * 문체 반영 모델(Accuracy): 22.8%
  * 문체 미반영 모델(Accuracy): 04.1%
  * Precision, Recall, F1-score를 통해 감성 분석 성능 평가


    ```
    # Trainer 정의 및 학습
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    ```

4. 문체 연관 감성 분석 결과


- 학습모델
* 평균 BLEU Score: 0.015~0.4
* 감성 분석 분포:
* Positive: 1건
* Negative: 2건


- 비학습모델
* 평균 BLEU Score: 0.015~0.2
* 감성 분석 분포:
* Positive: 1건
* Negative: 2건


<br/>


## 🔍 주요 실험 결과
### 1.문체 학습 모델(GPT-2)
  - 프롬프트	생성 텍스트 (Generated Text)	평균 BLEU Score: 0.0657

--------------------------------------------------
--- Prompt: We ---


Generated Text: We have no doubt about that, but....


BLEU Score: 0.7977


--------------------------------------------------
--- Prompt: The day was bright ---

Generated Text: The day was bright, the clouds were thick, ....


BLEU Score: 0.6921

--------------------------------------------------
--- Prompt: she ---


Generated Text: she, odin, my dear odin, I beg you to....


BLEU Score: 0.8452

--------------------------------------------------

### 2.학습 모델과 비학습 모델의 성능 비교

--------------------------------------------------
--- Comparison for Prompt 1 ---
Prompt: We

Trained Model:
Generated Text: We have seen your letter and all....


Sentiment: Positive


BLEU Score: 0.2287


Untrained Model:
Generated Text: We just hope you understand and respect the...


Sentiment: Negative


BLEU Score: 0.0410

--------------------------------------------------

--- Comparison for Prompt 2 ---
Prompt: The day was bright

Trained Model:
Generated Text: The day was bright, bright as ever,.....


Sentiment: Negative


BLEU Score: 0.0657






Untrained Model:
Generated Text: The day was bright, dark, and beautiful," he added...


Sentiment: Positive


BLEU Score: 0.0695

--------------------------------------------------

--- Comparison for Prompt 3 ---
Prompt: she

Trained Model:
Generated Text: she was only eleven years old.....


Sentiment: Negative


BLEU Score: 0.1968




Untrained Model:
Generated Text: she would go. The next morning,.....


Sentiment: Negative


BLEU Score: 0.1400


--------------------------------------------------


## 🛠 한계 및 개선 방향
### 한계
  - BLEU 점수가 일부 프롬프트에서 낮게 측정됨
    -   → 참조 문장 다양성 부족
  - 감성 분석 모델의 성능 저조
    - → 라벨링 및 데이터 품질 문제
  - 문체와 감성 간 관계를 명확히 입증하지 못함
    

### 개선 방향
  - 다양한 데이터셋 활용:
  - 데이터 증강을 통해 데이터 확보
  - 평가 지표 보완:
  - BLEU 외에 ROUGE, BERTScore 등 추가 지표 활용
  - 라벨링 개선 및 데이터 증강:
  - 감정 라벨링 고도화 및 불균형 문제 해결
<br/>

### 📚 참고문헌


Radford, A., et al., 2019. "Language Models are Few-Shot Learners," Advances in Neural Information Processing Systems.


Zhang, T., et al., 2021. "Style Example-Guided Text Generation using Generative Adversarial Transformers," Journal of Computational Linguistics.
<br/>


## 📝 실행 방법
필요한 라이브러리 설치:
- pip install datasets transformers textblob nltk


- Google Drive에서 데이터 로드 및 전처리.


- train.py를 실행하여 모델 학습 수행.


- generate.py로 텍스트 생성 및 BLEU 점수 계산.
