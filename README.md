# 🍩 Problem-solving-with-Donut

## 프로젝트 소개
- 이미지로 문제를 입력했을때 이미지 속 문제의 답을 얻을 수 있는 웹페이지 제작
- [Donut모델](https://github.com/clovaai/donut), openai api, streamlit을 이용해 프로젝트 진행
- Donut모델로 문제 이미지로부터 문제와 선지를 parsing하여 openai의 GPT-3.5 모델에 입력하여 답변 출력, streamlit으로 웹페이지 구현

![image](https://github.com/yeonsue/Problem-solving-with-Donut/assets/72684838/4483df37-cae4-4c21-b09c-2645990ed3ca)




## Donut model finetuning
- [dataset](https://drive.google.com/drive/folders/1JYtH3xyLS8vUvI5A0IXQO0vABurhRkmc?usp=sharing) : 토익 문제 300개, 자체 제작한 데이터셋으로 finetuning 진행
- [finetuned model](https://drive.google.com/drive/folders/1yl60PJnzVXkZdDOqgLWMqZpeAMDovYIe?usp=drive_link)

## Getting Started
- finetune-model 폴더에 [학습완료한 모델](https://drive.google.com/drive/folders/1yl60PJnzVXkZdDOqgLWMqZpeAMDovYIe?usp=drive_link) 추가
- 자기 openai api 키 .env file에 입력
```
    pip install .
    pip install timm==0.5.4
    pip install python-dotenv
    pip install langchain
    pip install openai
    pip install streamlit
```

  
## 'streamlit run Toeic.py' 입력해서 실행
결과 화면

![result](https://github.com/yeonsue/Problem-solving-with-Donut/assets/72684838/a7936b3e-6225-4e17-9981-d9ad041a1934)
