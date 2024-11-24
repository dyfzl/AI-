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
python
코드 복사

  #####    문체 보존 전처리 함수
  ```
  def clean_text_preserve_style(text):
      import re
      text = re.sub(r"[^\w\s.,!?']", "", text)  # 특수문자 유지
      text = re.sub(r"\s+", " ", text).strip()  # 공백 제거
      return text
  ```

  
2. 문체 학습 모델(GPT-2)
모델 구성:
  - Hugging Face의 GPT-2 모델을 사용하여 특정 작가(author=3)의 데이터를 학습
  - 학습 데이터(90%), 평가 데이터(10%)로 분할 후 Fine-tuning
  - 모델 학습:
  - 학습 손실이 0.401로 수렴
  - BLEU 점수를 사용하여 텍스트 생성 품질 평가
  - 텍스트 생성:
  - 프롬프트 "she"로 테스트한 결과, 학습된 문체를 반영한 문장을 생성
     - 예: Generated Text: she should have told you then all that was going on in the house.


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
  * 문체 반영 모델(Accuracy): 44.7%
  * 문체 미반영 모델(Accuracy): 44.0%
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

4. BLEU 점수 및 감성 분석 결과

   
  BLEU 점수:
* 평균 BLEU Score: 0.015~0.023
* 최고 BLEU Score: 0.9199 (프롬프트 "she")
* 감성 분석 분포:
* Positive: 2건
* Neutral: 1건
* Negative: 0건

##### BLEU 점수 계산

```
bleu_score = sentence_bleu([ref[0] for ref in references], candidate, weights=(0.5, 0.5), smoothing_function=smoothing)


print(f"BLEU Score: {bleu_score:.4f}")
```

<br/>


## 🔍 주요 실험 결과
### 문체 학습 모델(GPT-2)
  - 프롬프트	생성 텍스트 (Generated Text)	BLEU Score
  - "she"	she should have told you then all that was going on in the house.	0.9199
  - "The day was bright"	The day was bright. A terrible cloud of fire arose on the horizon.	0.0170
### 감성 분석(DistilBERT)
  - 모델	Accuracy	Precision	Recall	F1-Score
  - 문체 보존 모델	44.7%	43.9%	44.7%	43.5%
  - 문체 미반영 모델	44.0%	43.5%	44.0%	43.3%


##📈 시각화
1. BLEU 점수 히스토그램: 각 프롬프트에 대한 BLEU 점수 분포
2. 감성 분석 결과 바 차트:
Positive, Neutral, Negative 분포
3. 훈련 손실 그래프: Epoch별 Training Loss 감소 추이
<br/>

## 🛠 한계 및 개선 방향
*한계
  - BLEU 점수가 일부 프롬프트에서 낮게 측정됨 → 참조 문장 다양성 부족
  - 감성 분석 모델의 성능 저조 → 라벨링 및 데이터 품질 문제
  - 문체와 감성 간 관계를 명확히 입증하지 못함
  - 

*개선 방향
  - 다양한 데이터셋 활용:
  - 여러 작가의 데이터를 학습하여 모델 일반화
  - 평가 지표 보완:
  - BLEU 외에 ROUGE, BERTScore 등 추가 지표 활용
  - 라벨링 개선 및 데이터 증강:
  - 감성 라벨링 고도화 및 불균형 문제 해결
<br/>

##📚 참고문헌


Radford, A., et al., 2019. "Language Models are Few-Shot Learners," Advances in Neural Information Processing Systems.


Zhang, T., et al., 2021. "Style Example-Guided Text Generation using Generative Adversarial Transformers," Journal of Computational Linguistics.
<br/>


##📝 실행 방법
필요한 라이브러리 설치:
bash
코드 복사
pip install datasets transformers textblob nltk
Google Drive에서 데이터 로드 및 전처리.
train.py를 실행하여 모델 학습 수행.
generate.py로 텍스트 생성 및 BLEU 점수 계산.
결과 분석 및 시각화.
