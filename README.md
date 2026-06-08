# Tiny GPT Literature Generator

수업시간의 `notebook_06` 구조를 바탕으로 만든 **문자 단위(char-level) Tiny GPT** 프로젝트입니다.

목표는 `data/input.txt`에 들어 있는 문학 텍스트를 학습해서, 앞부분 prompt를 주면 그 뒤에 올 문자를 예측하며 새로운 문장을 생성하는 것입니다.


## 1. 폴더 구조
```text
├── README.md                          # 프로젝트 설명 및 실행 방법
├── requirements.txt                   # 필요한 패키지 목록, torch>=2.0
├── train.py                           # input.txt를 이용해 GPT 모델 학습
├── generate.py                        # 학습된 모델로 문장 생성
├── model.py                           # Tiny GPT 모델 구조 정의
├── tiny_gpt.ipynb                     # 수업 notebook_06 스타일의 노트북
├── data/
│   └── input.txt                      # 학습할 소설 텍스트 입력 파일
└── checkpoints/
    └── .gitkeep                       # 모델 저장 폴더 유지용 파일
```



## 2. 사용법


- 학습
```bash
python train.py
```

빠르게 테스트만 하고 싶을 때:

```bash
python train.py --max_iters 100 --eval_interval 20 --batch_size 16 --n_embd 64 --n_layer 2 --n_head 2
```

학습이 끝나면 `checkpoints/tiny_gpt.pt`가 생성됩니다.

- Sampling 
```bash
python generate.py --prompt "인력거" --max_new_tokens 500
```


## 3. 모델 구조

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


## 4. 실행
### 4-1. 현진건 - 운수좋은날 (prompt:"인력거"  max_new_tokens:500)
- 학습
```text
text length: 10,340
vocab size: 687
parameters: 976,559
step    0 | train loss 6.4018
step  100 | train loss 4.1088
step  200 | train loss 3.6087
step  300 | train loss 3.2997
step  400 | train loss 3.0509
step  500 | train loss 2.8581
step  600 | train loss 2.6566
step  700 | train loss 2.4423
step  800 | train loss 2.1880
step  900 | train loss 1.8803
step  999 | train loss 1.6223
```

- 1000회 학습 후 sampling
```text
인력거주를 노나는 하는 듯방한 무엇 채워그란 가랄데 연해 워졌다. 그보다 남편이란 웃음질이 염
괴상건만 처박게 점이 되었다. 오늘어이 고개를 김첨지는 얼굴을 변이 기도록 엄습해 거기도 교(음과 기
이든 마음질렀다. 거기데 그 인력거기운 동안 노릇을 하았다겠으로 떠서 타고 늦추기도 같이니와 학교 주었다. 고 해 전차기가기도 이니 이때에 놓였지니
무슨 몸을 병을 제 총총히 옷에서 뻐천방지 많은 알아 비
의 이 눈은 머리고 같은 버이 듯하고 모든니만에서 또 볼을 비는 내어 듯이 하고 불쑥 듯하고 같은 애쓰다가슴이 좁쌀 다.
니 가까운 까 하였다구두한 정적거뿐이다. 오로 전재미를 보고 전짜고 있다. 그 온
치삼이는 얼마며 엉 자주에서 이라 얼굴을 내려 기는 어서 조한번 찼다. “일이 산 랍시켰다.”
라고 십 김첨지고개를 치뜬 땀이 적거리가. 전차가 까지 저히 오래 사도 비롯이 돈이 사면 돈! 이다. 내 빨 돈은
눈을 김첨지는 이 어린애쓰다 맞붙여 “그의 신조금 오는 썩은 보는 고함적실하였다. 제
```


- 2000회 학습 후 sampling
```text
인력거를 털털털거리며 이 우중에 돌아가지고리게, 몰랐다. 자식어디방
리는 메였다. 구해 간 제 지는 에게 속히어진다 나기 있게 소리를 내나
다 하더 듯이 팔십 내리는 내기운 하나기쁨을 하더니만 일 꼬리 원 앞 어린다과 것 오늘 펄쩍 집어던졌다. 그 사품에 은
이에 몇 - 떨어진이 팔시며 쨍 내가 돈을 끓이며 돈을 화를 떨어 육십 전이란 돈을 눌며 콩팥이 육
에 쥠으며 인력거를 세인력거주저으면 안과 뚝 한 체한 간답이 한 점이도 있고 다. 김첨지의 또 입에서 떨어졌다. 제 입으로 부
르고도 스스로 그 엄청난 돈 액을 곳을 벌었을 벌었어,
“예, 이 타시를 사라치겠구 안됐데.”
라고 우쳤다.
“뭐, 이 마누라가 죽어서 양으니까 김첨지
를 뒤축으랴.
“설렁하고 또 인력거를 묵을 뿌리고 근슬근 잡아 나서 듯이 하였
였다. 그의 얼굴은 유달으며 질팡른거리고 움찐 기
이 비는 오히려 찌할 알아니 적추어지고 제 다가 술은 아내의 행운을 붓다고 내저나가 듯이 어석에게 아니고 또 또 제 얼굴을 죽은 말리
```

### 4-2. 윤동주 - 별 헤는 밤 (prompt:"별 헤는 밤"  max_new_tokens:300)
- 학습
```text
text length: 659
vocab size: 187
parameters: 848,059
step    0 | train loss 5.1657
step  100 | train loss 2.1800
step  200 | train loss 1.0083
step  300 | train loss 0.3780
step  400 | train loss 0.1544
step  500 | train loss 0.0913
step  600 | train loss 0.0698
step  700 | train loss 0.0589
step  800 | train loss 0.0534
step  900 | train loss 0.0501
step  999 | train loss 0.0477
```
- 500회 학습 후 sampling
```text
별 헤는 밤
가을로 하늘에는것은
쉬이 오는 까닭입니다.

쉬이 나는 아무 걱정도 헤일듯이름자 써보고,
흙으로 덮어 버십니다.

따는 걱정도 풀이 당신은
부끄러운 이름을 너무 슬퍼하는 의 까닭입니다.

그러나 小學校 겨울이 지나고 나의 별과
별에도 하나에속에 봄이 오면
무덤 • 우에 파란 잔디가 피어머니다.

(1. 피어나듯이 함과
島에 멀리었습니다.

따는 밤을 새겨지는 까닭이오면
무憬과
가 밤이 멀리고 北間島에 계십니다.
나는 무엇인지 그리워
이 많은 별빛이 나린 무성할게성할게같이름자를 많은 린 언덕우에 언덕우에이름자를 써보고,
흙으로 덮어
```

- 1000회 학습 후 sampling
```text
별 헤는 밤

季節이 지나가는 하늘에는
가을로 가득 차있읍니다.

나는 아무 걱정도 없이
가을 속의 별들을 다 헤일듯합니다.

가슴속에 하나 둘 새겨지는 별을
이제 다 못헤는것은
쉬이 아츰이 오는 까닭이오,
來日 밤을 새워 우에 파란 잔디가 피어나듯이
내 이름자 묻힌 언덕우에도
자랑처럼 풀이 무성할게덤 (1941십니다. 小學校 때 冊床을 우에 같이 했든 아이들의 이름과, 佩, 鏡, 玉 이런 異國少女들의 이름과 벌써 애기 어머니 된 계집애들의 이름과, 가난한 이웃사람들의 이름과, 비둘기, 강아지, 토끼, 노새, 노루, “푸랑시스 • 쨤”, “라
```

### 4-3 . tinyshakespeare (prompt:"Roemo:"  max_new_tokens:500)
- 학습
```text
text length: 1,115,393
vocab size: 65
parameters: 816,705
step    0 | train loss 4.2304
step  100 | train loss 2.6535
step  200 | train loss 2.5099
step  300 | train loss 2.4353
step  400 | train loss 2.3572
step  500 | train loss 2.2931
step  600 | train loss 2.2345
step  700 | train loss 2.1760
step  800 | train loss 2.1258
step  900 | train loss 2.0821
step  999 | train loss 2.0446
```

-1000회 학습 후 sampling
```text
Romeo: how cold for und agelt.

Guenkermy to thimo mame kimaly,
Asperof be thall he fore!

QUTAI:
Am cow scae.

TISTAR:
And pay, noar of you widext, ling, with spime:
No not stie to in that thisesir wil hencting ures,
That: ng, a croyever her the wi harm, the you kin.

BOMELONENUS:
Howare, you. if dee ond God comee quer to
Wher that Youn the prosene ther swere sult ping'd
The with the hearist he be in to I thise of of the breas't,
Farcat mermion ther you ler.

GRICET?
MA:

Ocha3ll, knot, my and the to
```

-2000회 학습 후 sampling
```text
Romeo:
We whonk sir, lord hrim syour the platio?

FirTIS ELINCE:
I ellaccle, wost beard Weath in sowa come and
You dofe them his ind the havistled's best as a
Se Marfurwoud one the as was measi's father
And thou wort'st and bord Andows;
Marry, mood I conasto mill thou best cure
Mess not batting your he; bose probald the cappleaps;
And a wian he the neve my that eyembery.
I Ellaam we is spuce, ago, I soun he hear lovave mone
And is a fale.
Angainte you and at like and my nond to no pervides.

HENRY's
```
