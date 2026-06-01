# AM234 Report — 完整消融實驗報告（v6_ctrl / v6_a / v6_b / v6 / v7）

## 任務背景

**輸入**：一條 1000bp 的 DNA 序列  
**輸出**：919 個二元標籤（每個 chromatin feature 是否存在）  
**Batch size**：256（每次同時處理 256 條序列）  
**資料集**：Train 4,400,000 條 / Valid 8,000 條 / Test 455,024 條

---

## 第一步：DNA → One-Hot 編碼（所有模型共用）

DNA 只有四種鹼基：A、C、G、T。

One-hot 編碼是把每個鹼基轉成一個 4 維向量，規則如下：

```
A → [1, 0, 0, 0]
C → [0, 1, 0, 0]
G → [0, 0, 1, 0]
T → [0, 0, 0, 1]
```

所以一條 1000bp 的序列，轉成 tensor 後是：

```
形狀：[4, 1000]
含義：4 個鹼基 × 1000 個位置

例子（前3個位置 = ACG）：
位置：  0    1    2   ...  999
      [1,   0,   0,  ...  ]   ← A 那一列
      [0,   1,   0,  ...  ]   ← C 那一列
      [0,   0,   1,  ...  ]   ← G 那一列
      [0,   0,   0,  ...  ]   ← T 那一列
```

加上 batch 維度後，輸入 tensor 為 **[B, 4, 1000]** = [256, 4, 1000]。

---

## 什麼是「升維」？

**升維（dimension upgrade）** 是把每個位置的向量從低維（4-dim）變成高維（128-dim）。

### 為什麼需要升維？

4-dim one-hot 的問題：
- 每個位置只有 4 種可能的向量（A、C、G、T 四選一）
- 向量之間沒有「距離」或「相似度」的概念（A 和 C 一樣陌生）
- 無法表達「這個位置加上它的鄰居共同代表什麼意義」

升到 128-dim 之後：
- 每個位置有一個連續的 128 維向量
- 向量空間中的位置有語意（相似的序列 context → 相近的向量）
- 128 維的向量可以被 attention 機制用來計算「相關性」

### 升維的三種方式

| 模型 | 升維方式 | 信息量 |
|------|----------|--------|
| v6_ctrl | **不升維**，直接用 4-dim one-hot | 4 種可能值 |
| v6_b | **不升維**，直接用 4-dim one-hot | 4 種可能值 |
| v6_a | **Linear(4→128)**，矩陣乘法 | 仍然只有 4 種可能值（只是放大） |
| v6 | **CausalConv(k=5)**，卷積 | 4^5 = 1024 種可能值（5-mer context）|
| v7 | **CausalConv(k=5)**，卷積 | 4^5 = 1024 種可能值（5-mer context）|

---

## v6_ctrl 完整流程

### 架構圖

```
[B, 4, 1000]
    │
    │ 不升維，直接分割
    ▼
reshape → [B, 20, 50, 4]     ← 把 1000bp 切成 20 個 50bp 的 chunk
    │
    │ Self-Attention（每個 chunk 獨立）
    ▼
[B, 20, 50, 128]             ← 每個 chunk 的每個位置有 128-dim 表示
    │
    │ Max-pool（取最重要的位置）
    ▼
[B, 20, 128]                 ← 每個 chunk 濃縮成一個 128-dim 向量
    │
    │ Linear(128 → 919)
    ▼
[B, 20, 919]                 ← 每個 chunk 對 919 個 track 的預測貢獻
    │
    │ Dense Readout（mean-pool + MLP）
    ▼
[B, 919]                     ← 最終預測：919 個 logit
```

### Step 1：reshape — 把序列切成 chunk

```python
h = x.transpose(1, 2)            # [B, 4, 1000] → [B, 1000, 4]
h = h.reshape(B, 20, 50, 4)      # [B, 1000, 4] → [B, 20, 50, 4]
```

把 1000 個位置重新排列成 20 個 chunk，每個 chunk 50 個位置：

```
原序列（1000bp）：
[pos0, pos1, ..., pos49, pos50, ..., pos99, ..., pos999]

分成 20 個 chunk（每個 50bp）：
chunk0  : [pos0   ~ pos49 ]
chunk1  : [pos50  ~ pos99 ]
chunk2  : [pos100 ~ pos149]
...
chunk19 : [pos950 ~ pos999]
```

此時每個位置仍然是 4-dim one-hot（沒有升維）。

### Step 2：Self-Attention — 在 chunk 內部計算注意力

這是最核心的步驟。在每個 50bp 的 chunk 內部，讓每個位置去「問」其他位置：**你跟我有多相關？**

#### 2-1：計算 Q、K、V（Query、Key、Value）

```python
W_q = nn.Linear(4, 128, bias=False)   # 可訓練矩陣，shape [128, 4]
W_k = nn.Linear(4, 128, bias=False)   # 可訓練矩陣，shape [128, 4]
W_v = nn.Linear(4, 128, bias=False)   # 可訓練矩陣，shape [128, 4]

Q = W_q(h)   # [B, 20, 50, 4] → [B, 20, 50, 128]
K = W_k(h)   # [B, 20, 50, 4] → [B, 20, 50, 128]
V = W_v(h)   # [B, 20, 50, 4] → [B, 20, 50, 128]
```

**Q（Query）**：「我想找什麼」—— 每個位置的「問題」  
**K（Key）**：「我是什麼」—— 每個位置的「標籤」  
**V（Value）**：「我的內容」—— 每個位置的「資訊」

這裡的 Linear(4→128) 是矩陣乘法：把每個 4-dim 向量乘以一個 [128×4] 的矩陣，得到 128-dim 向量。Q、K、V 三個矩陣完全獨立，各自學習不同的投影方式。

> **注意**：v6_ctrl 的「升維」發生在這裡（4→128），但 Q、K、V 是三個獨立的投影，每個位置的輸入仍然只有 4 種可能，所以本質上信息量沒有增加。

#### 2-2：計算 Attention Score（相關性分數）

```python
scores = Q @ K.transpose(-1, -2) * (1 / sqrt(128))
# Q shape:  [B, 20, 50, 128]
# K.T shape:[B, 20, 128, 50]
# 結果:     [B, 20, 50, 50]
```

對於 chunk 內的每一對位置 (i, j)，計算：

```
score(i, j) = Q[i] · K[j] / sqrt(128)
```

這是一個點積（dot product）—— 向量越相似，分數越高。  
除以 sqrt(128) 是防止數值太大導致 softmax 梯度消失（scaling trick）。

結果是一個 **50×50 的矩陣**，代表 chunk 內所有位置對之間的相關性：

```
          pos0  pos1  pos2  ...  pos49
pos0    [ 0.8,  0.1,  0.3, ...,  0.2 ]
pos1    [ 0.1,  0.9,  0.2, ...,  0.1 ]
pos2    [ 0.3,  0.2,  0.7, ...,  0.4 ]
...
pos49   [ 0.2,  0.1,  0.4, ...,  0.8 ]
```

#### 2-3：Softmax — 把分數轉成機率分佈

```python
A = softmax(scores, dim=-1)
# shape: [B, 20, 50, 50]
```

對每個位置 i 的那一行做 softmax，讓 50 個分數加總為 1：

```
pos0 的注意力分佈：
[0.40,  0.05,  0.15, ...,  0.10]   ← 加總 = 1.0
  ↑
pos0 把 40% 的注意力放在自己身上
```

#### 2-4：加權求和 Values

```python
out = A @ V
# A shape: [B, 20, 50, 50]
# V shape: [B, 20, 50, 128]
# 結果:    [B, 20, 50, 128]
```

每個位置 i 的輸出 = 以 attention 為權重，對所有位置的 V 做加權平均：

```
output[i] = Σ_j  A[i,j] × V[j]
           = A[i,0]×V[0] + A[i,1]×V[1] + ... + A[i,49]×V[49]
```

結果：每個位置得到一個新的 128-dim 向量，**融合了 chunk 內其他所有位置的信息**，融合比例由 attention 決定。

#### 2-5：Max-Pool — 取 chunk 的代表向量

```python
out = out.max(dim=2).values
# [B, 20, 50, 128] → [B, 20, 128]
```

在每個 chunk 的 50 個位置中，對每個維度取最大值，得到該 chunk 的「最顯著特徵」。

### Step 3：Linear Projection → 919 tracks

```python
proj = nn.Linear(128, 919)
out = proj(out)    # [B, 20, 128] → [B, 20, 919]
```

把每個 chunk 的 128-dim 表示，投影成 919-dim，代表這個 chunk 對 919 個 chromatin track 各自的貢獻。

### Step 4：Dense Readout — 整合所有 chunk，輸出預測

```python
# mean-pool
x_mean = out.mean(dim=1)           # [B, 20, 919] → [B, 919]

# 2-layer MLP
out = Linear(919, 919)(x_mean)
out = BatchNorm1d(919)(out)
out = ReLU()(out)
out = Linear(919, 919)(out)        # → [B, 919]
```

把 20 個 chunk 的預測平均，再過兩層 MLP 做非線性組合，得到最終 919 個 logit。

---

## v6_a 完整流程

### 架構圖

```
[B, 4, 1000]
    │
    │ Linear(4→128)，每個位置獨立升維
    ▼
[B, 1000, 128]               ← 每個位置有 128-dim，但沒有鄰居信息
    │
    │ reshape
    ▼
[B, 20, 50, 128]             ← 切成 20 個 chunk，每 chunk 50 個位置
    │
    │ Self-Attention（chunk 內部）
    ▼
[B, 20, 50, 128]
    │
    │ Max-pool
    ▼
[B, 20, 128]
    │
    │ Linear(128 → 919)
    ▼
[B, 20, 919]
    │
    │ Dense Readout
    ▼
[B, 919]
```

### 和 v6_ctrl 的差異：先做 Linear(4→128)

```python
embed = nn.Linear(4, 128, bias=False)   # 共用的升維矩陣，shape [128, 4]

h = x.transpose(1, 2)        # [B, 4, 1000] → [B, 1000, 4]
h = embed(h)                  # [B, 1000, 4] → [B, 1000, 128]
h = h.reshape(B, 20, 50, 128) # → [B, 20, 50, 128]
```

然後 Q、K、V 也從 128-dim 出發：

```python
W_q = nn.Linear(128, 128)   # 現在是 128→128
W_k = nn.Linear(128, 128)
W_v = nn.Linear(128, 128)
```

### 為什麼 v6_a 和 v6_ctrl 結果差不多？

**v6_a 的根本問題**：`Linear(4→128)` 的輸入只有 4 種可能的值。

```
A → embed(A) = embed_matrix 的第 0 列（永遠相同）
C → embed(C) = embed_matrix 的第 1 列（永遠相同）
G → embed(G) = embed_matrix 的第 2 列（永遠相同）
T → embed(T) = embed_matrix 的第 3 列（永遠相同）
```

不管 embed_matrix 有多少維（128 維），輸出永遠只有 **4 種可能的 128-dim 向量**。每個位置的 embedding 只取決於它是 A/C/G/T，**完全不知道鄰居是誰**。Attention 之後雖然融合了鄰居信息，但初始 embedding 太貧乏，效果和 v6_ctrl 幾乎一樣。

---

## v6 完整流程

### 架構圖

```
[B, 4, 1000]
    │
    │ CausalConvEmbed(k=5)：看左邊 4 個鄰居 + 自己
    ▼
[B, 128, 1000]               ← 每個位置有 128-dim，包含 5-mer 上下文
    │
    │ transpose + reshape（切法不同！50 window × 20 位置）
    ▼
[B, 50, 20, 128]             ← 切成 50 個 window，每 window 20 個位置
    │
    │ Self-Attention（window 內部）
    ▼
[B, 50, 20, 128]
    │
    │ Max-pool
    ▼
[B, 50, 128]
    │
    │ Linear(128 → 919)
    ▼
[B, 50, 919]
    │
    │ Dense Readout
    ▼
[B, 919]
```

### Step 1：CausalConvEmbed — 看鄰居升維

```python
# 左側補 (k-1)=4 個零，右側不補
h = F.pad(x, (4, 0))        # [B, 4, 1000] → [B, 4, 1004]

# Conv1d(in_channels=4, out_channels=128, kernel_size=5, padding=0)
h = conv(h)                  # [B, 4, 1004] → [B, 128, 1000]
```

**為什麼叫 Causal（因果）？**

普通卷積（左右各補 2）：位置 i 看 [i-2, i-1, i, i+1, i+2]（看未來）  
因果卷積（只補左側 4 個零）：位置 i 看 [i-4, i-3, i-2, i-1, i]（只看過去）

```
Causal（左向）：
  ... [i-4][i-3][i-2][i-1][ i ]   ← 只看過去 4 個鄰居
                               ↑
                          當前位置
```

每個 kernel_size=5 的卷積核，接觸 5 個連續位置的 4-dim one-hot，合計 5×4=20 個輸入，輸出 1 個值。128 個 filter 輸出 128-dim 向量。

**5-mer 信息量**：5 個位置各有 4 種鹼基 → 4^5 = **1024 種可能的 5-mer**，比 4 種 one-hot 多 256 倍。

升維後每個位置的 128-dim 向量 **代表的是「以當前位置為結尾的 5-mer 是什麼」**，而不只是「這個位置是什麼鹼基」。

### Step 2：reshape — 切法和 v6_ctrl 不同

```python
h = h.transpose(1, 2)          # [B, 128, 1000] → [B, 1000, 128]
h = h.reshape(B, 50, 20, 128)  # → [B, 50, 20, 128]
```

**v6_ctrl 的切法**：20 個 chunk，每 chunk 50 個位置
```
chunk0: pos0~49,  chunk1: pos50~99, ..., chunk19: pos950~999
每個 chunk 跨度 = 50bp
```

**v6 的切法**：50 個 window，每 window 20 個位置
```
window0: pos0~19,  window1: pos20~39, ..., window49: pos980~999
每個 window 跨度 = 20bp
```

### Step 3：Self-Attention — 在 20bp window 內計算注意力

```python
Q = W_q(h)   # [B, 50, 20, 128] → [B, 50, 20, 128]
K = W_k(h)
V = W_v(h)

scores = Q @ K.transpose(-1, -2) / sqrt(128)   # [B, 50, 20, 20]
A = softmax(scores, dim=-1)                     # [B, 50, 20, 20]
out = A @ V                                     # [B, 50, 20, 128]
```

Attention score matrix 是 **20×20**（v6_ctrl 是 50×50）。

**20bp window 為什麼更有效？**

TF（轉錄因子）binding motif 的典型長度是 6–20bp。一個 20bp window 剛好可以完整包含一個 binding site：

```
一個 20bp window 中可能包含一個完整的 CTCF 結合位點（~19bp）：
pos0 pos1 pos2 ... pos18 pos19
 C    C    G  ...   C    G     ← CTCF consensus

Attention 學到：
pos2(G) 和 pos7(G) 和 pos12(G) 同時出現 → 這是 CTCF 位點
```

v6_ctrl 的 50bp window 中，binding site 只佔 12/50 = 24%，信噪比更低，attention 更難聚焦。

### Step 4：Max-pool + 投影

```python
out = out.max(dim=2).values    # [B, 50, 20, 128] → [B, 50, 128]
out = proj(out)                 # [B, 50, 128] → [B, 50, 919]
```

### Step 5：Dense Readout

```python
out = out.mean(dim=1)          # [B, 50, 919] → [B, 919]
# Linear → BN → ReLU → Linear
out = mlp(out)                  # → [B, 919]
```

---

## 所有模型關鍵差異對比

| 模型 | 升維方式 | Window | Attn levels | val AUROC | test AUROC |
|------|----------|--------|-------------|-----------|------------|
| v6_ctrl | 無（4-dim） | 50bp | 1層 | 0.707 | 0.711 |
| v6_a | Linear(4→128) | 50bp | 1層 | 0.703 | 0.699 |
| v6_b | 無（4-dim） | 20bp | 1層 | 0.702 | 0.705 |
| v6 | CausalConv(k=5) | 20bp | 1層 | 0.786 | 0.779 |
| **v7** | CausalConv(k=5) | 20bp | **2層** | **0.797** | **0.787** |

---

## Attention 過程完整數學（以 v6 單一 window 為例）

```
輸入：h ∈ R^{20 × 128}    （20 個位置，每個 128-dim，已含 5-mer 上下文）

1. 計算 Q, K, V：
   Q = h × W_q^T  ∈ R^{20 × 128}
   K = h × W_k^T  ∈ R^{20 × 128}
   V = h × W_v^T  ∈ R^{20 × 128}

2. 計算相關性分數：
   S = Q × K^T / √128  ∈ R^{20 × 20}
   S[i,j] = 位置 i 對位置 j 的原始關注度

3. Softmax（每行獨立）：
   A = softmax(S, dim=-1)  ∈ R^{20 × 20}
   A[i,j] = 位置 i 分配給位置 j 的注意力比例（Σ_j A[i,j] = 1）

4. 加權聚合：
   out = A × V  ∈ R^{20 × 128}
   out[i] = Σ_j  A[i,j] × V[j]
          = 以 attention 為權重，聚合所有位置的 Value

5. Max-pool（取最重要的位置）：
   final = max_{i=0..19}(out[i])  ∈ R^{128}
```

---

## v6_b 架構（新增消融）

v6_b 是專門用來隔離「window size」效果的模型，其他設計和 v6_ctrl 完全相同，唯一差異是把 50bp window 改成 20bp：

```
[B, 4, 1000]
    │
    │ 不升維，直接分割（切法不同）
    ▼
reshape → [B, 50, 20, 4]     ← 50 個 20bp 的 window（v6_ctrl 是 20 個 50bp）
    │
    │ Self-Attention（window 內部，20×20 matrix）
    ▼
[B, 50, 20, 128]
    │
    │ Max-pool
    ▼
[B, 50, 128]
    │
    │ Linear(128 → 919) + Dense Readout
    ▼
[B, 919]
```

```python
h = x.transpose(1, 2).reshape(B, 50, 20, 4)   # 50 window × 20bp，raw 4-dim
Q, K, V = W_q(h), W_k(h), W_v(h)              # W_q: Linear(4→128)
A = softmax(Q @ K.T / sqrt(128), dim=-1)       # [B, 50, 20, 20]
out = (A @ V).max(dim=2).values                # [B, 50, 128]
```

**結果：val AUROC = 0.702**，和 v6_ctrl（0.707）幾乎相同。

---

## v7 架構（真正的 Double Attention）

v6 的名字叫「double attn」但其實只有一層 attention，第二層是 mean-pool。v7 才是真正的兩層 attention：

```
[B, 4, 1000]
    │
    │ CausalConvEmbed(k=5)
    ▼
[B, 128, 1000]
    │
    │ reshape → [B, 50, 20, 128]
    ▼
Level 1：intra-window attention（20×20，在 20bp 內部）
    │
    │ max-pool
    ▼
[B, 50, 128]                 ← 每個 20bp window 的代表向量
    │
    │ Level 2：inter-window attention（50×50，在 50 個 window 之間）
    ▼
[B, 50, 128]                 ← 每個 window 已融合其他 window 的資訊
    │
    │ Linear(128 → 919) + Dense Readout
    ▼
[B, 919]
```

### Level 1：intra-window attention（和 v6 相同）

```python
embed = CausalConvEmbed(k=5)
h = embed(x).transpose(1, 2)          # [B, 1000, 128]
h = h.reshape(B, 50, 20, 128)         # [B, 50, 20, 128]

Q1, K1, V1 = W_q1(h), W_k1(h), W_v1(h)
A1 = softmax(Q1 @ K1.T / sqrt(128), dim=-1)   # [B, 50, 20, 20]
h = (A1 @ V1).max(dim=2).values               # [B, 50, 128]
```

每個 20bp window 內部做 self-attention，max-pool 後得到 50 個 window 的代表向量。

### Level 2：inter-window attention（v7 新增）

```python
Q2, K2, V2 = W_q2(h), W_k2(h), W_v2(h)
A2 = softmax(Q2 @ K2.T / sqrt(128), dim=-1)   # [B, 50, 50]
h = A2 @ V2                                   # [B, 50, 128]
```

50 個 window 之間互相做 attention：

```
          win0  win1  win2  ...  win49
win0    [ 0.3,  0.1,  0.4, ...,  0.2 ]   ← win0 關注哪些其他 window
win1    [ 0.1,  0.8,  0.1, ...,  0.0 ]
...
win49   [ 0.2,  0.0,  0.3, ...,  0.5 ]
```

**這一層學到的是**：「第 3 個 window（pos60~79）和第 27 個 window（pos540~559）同時有某種 pattern → 才能判定這個 chromatin track 存在」。也就是 1000bp 序列中不同區域之間的**長程相互依賴關係**。

**結果：val AUROC = 0.797**，比 v6（0.786）高 +1.1%。

---

## 完整消融實驗結論

消融設計（逐一隔離每個因素）：

```
┌──────────────────────────────────────────────────────┐
│  v6_ctrl (0.707)：50bp window，無 context，1層 attn   │
│                                                      │
│  ↓ 只換升維方式（Linear 4→128）                       │
│  v6_a (0.703)：差異 -0.004（無效）                    │
│  結論：升維本身無效，因為仍只有 4 種可能值              │
│                                                      │
│  ↓ 只換 window 大小（50bp → 20bp）                    │
│  v6_b (0.702)：差異 -0.005（無效）                    │
│  結論：window 縮小本身無效，沒有 context 支撐就沒用    │
│                                                      │
│  ↓ 同時換 window（20bp）+ 升維方式（CausalConv k=5）  │
│  v6 (0.786)：差異 +0.079（+11.2%）                   │
│  結論：CausalConv 帶來的 5-mer context 是主驅動力     │
│        20bp window 配合 5-mer context 才有意義        │
│                                                      │
│  ↓ 加第二層 inter-window attention                   │
│  v7 (0.797)：差異 +0.011（+1.4%）                    │
│  結論：跨 window 的長程信息確實有額外貢獻              │
└──────────────────────────────────────────────────────┘
```

### 三條核心結論

**結論 1：CausalConv k=5 context 是主驅動力（+11.2%）**

v6_b 排除了「window 大小」的解釋——光縮小到 20bp 沒有用（0.702 ≈ v6_ctrl 0.707）。
真正有效的是 CausalConv 給每個位置帶來的 5-mer 上下文（1024 種 vs 4 種可能值）。

**結論 2：20bp window 配合 5-mer context 才有意義**

如果 context 足夠豐富（5-mer），20bp window 的大小剛好對準 TF binding motif（6–20bp）的生物尺度，讓 attention 能精確識別 motif 的位置組合。兩者缺一不可。

**結論 3：Inter-window attention 有額外貢獻（+1.1%）**

v7 在 v6 的基礎上加了跨 window 的 attention，讓模型能捕捉 1000bp 範圍內不同區域的長程依賴。這個貢獻較小（+1.1%），但穩定存在，說明 chromatin state 的判定不僅看局部 motif，也看多個 motif 的組合關係。

### 參考基準

| Model | val AUROC | test AUROC | 備註 |
|-------|-----------|------------|------|
| Dense baseline（6層 bidir. transformer） | ~0.860 | — | 全序列 attention，5.6M params |
| v4（conv stem + 2-layer LIF） | **0.818** | **0.797** | DeepSEA-style conv |
| **v7（真 double attn）** | **0.797** | **0.787** | 本系列最佳 attention 模型 |
| v6（single attn + causal conv） | 0.786 | 0.779 | |
| v6_ctrl / v6_b / v6_a | ~0.703 | ~0.707 | 無 5-mer context 的上限 |

**生物學解釋**：TF binding motif 長 6–20bp，DNA 有強烈的局部 k-mer 組成偏好（CpG、AT-rich regions 等）。CausalConv(k=5) 的設計對準了這兩個生物學事實：每個位置的嵌入包含當前 5-mer 的身份，而 20bp window 的 attention 讓模型學習 motif 內部哪幾個 5-mer 位置共同出現才構成一個完整的 binding site。
