# Tiny GPT Literature Generator

수업시간의 `notebook_06` 구조를 바탕으로 만든 **문자 단위(char-level) Tiny GPT** 프로젝트입니다.

목표는 `data/input.txt`에 들어 있는 문학 텍스트를 학습해서, 앞부분 prompt를 주면 그 뒤에 올 문자를 예측하며 새로운 문장을 생성하는 것입니다.

기본 예시는 저작권 문제가 적은 **Shakespeare의 Romeo and Juliet 일부 문장**을 사용합니다. 황순원의 「소나기」 전문은 저작권이 남아 있을 수 있으므로 이 repository에는 포함하지 않았습니다. 과제에서 「소나기」를 쓰려면 본인이 합법적으로 사용할 수 있는 텍스트를 `data/input.txt`에 직접 넣으면 됩니다.

## 폴더 구조

```text
tiny-gpt-literature/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py
├── generate.py
├── model.py
├── notebook_06_style_tiny_gpt.ipynb
├── data/
│   └── input.txt
└── checkpoints/
    └── .gitkeep
```

## 1. 설치

```bash
pip install -r requirements.txt
```

## 2. 학습

기본 설정으로 학습:

```bash
python train.py
```

빠르게 테스트만 하고 싶을 때:

```bash
python train.py --max_iters 100 --eval_interval 20 --batch_size 16 --n_embd 64 --n_layer 2 --n_head 2
```

학습이 끝나면 `checkpoints/tiny_gpt.pt`가 생성됩니다.

## 3. 생성

```bash
python generate.py --prompt "ROMEO:" --max_new_tokens 500
```

한국어 텍스트로 학습했다면 예를 들어:

```bash
python generate.py --prompt "소년은" --max_new_tokens 500
```

## 4. 다른 텍스트로 바꾸는 법

`data/input.txt` 내용을 원하는 텍스트로 바꾼 뒤 다시 학습하면 됩니다.

예시:

```bash
# data/input.txt 안에 로미오와 줄리엣 또는 합법적으로 사용할 수 있는 소나기 텍스트를 붙여넣기
python train.py
python generate.py --prompt "ROMEO:"
```

## 모델 구조

`notebook_06`에서 배운 흐름과 거의 같습니다.

1. 텍스트를 문자 단위 vocabulary로 변환
2. `NextTokenDataset`으로 입력 `x`와 정답 `y` 생성
3. token embedding + position embedding
4. masked multi-head self-attention
5. feedforward network
6. residual connection
7. layer normalization
8. 여러 개의 Transformer block 쌓기
9. 다음 문자를 예측하는 language model head
10. sampling으로 새 텍스트 생성

## 주의

이 모델은 과제용 tiny GPT입니다. ChatGPT 같은 대형 모델이 아니라, 수업 내용 이해를 보여주기 위한 작은 character-level Transformer입니다.
