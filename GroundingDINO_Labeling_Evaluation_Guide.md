# GroundingDINO 輔助標注評估指導手冊

> **適用場景**：工業焊接工站目標檢測，以 GroundingDINO 作為半自動標注輔助工具，替代純人工標注流程，提升資料集建立效率。
>
> **目標讀者**：技術主管、開發工程師
>
> **核心主張**：在不顯著降低下游模型品質的前提下，透過 GroundingDINO 降低新工站的標注成本。

---

## 目錄

1. [整體評估框架](#1-整體評估框架)
2. [資料集規劃與切分策略](#2-資料集規劃與切分策略)
3. [GroundingDINO Fine-tune 流程](#3-groundingdino-fine-tune-流程)
4. [實驗一：標注品質驗證](#4-實驗一標注品質驗證)
5. [實驗二：標注效率比較](#5-實驗二標注效率比較)
6. [實驗三：跨工站泛化性比較](#6-實驗三跨工站泛化性比較)
7. [實驗四：下游模型品質驗證](#7-實驗四下游模型品質驗證)
8. [驗證報告撰寫結構](#8-驗證報告撰寫結構)
9. [常見問題與主管答辯準備](#9-常見問題與主管答辯準備)
10. [附錄：指標計算說明](#10-附錄指標計算說明)

---

## 1. 整體評估框架

### 1.1 核心敘事

本次評估的核心命題：

> 導入 GroundingDINO 作為標注輔助工具，在下游 YOLO 模型品質損失不超過 5% 的前提下，將新工站標注所需的純人工時間降低 50% 以上。

所有實驗設計均服務於此命題，技術指標是支撐，效率數據是結論。

### 1.2 四個實驗層次

```
實驗一：標注品質驗證
  └─ GroundingDINO 框出來的結果夠不夠好？（與 ground truth 比較）

實驗二：標注效率比較
  └─ 導入後實際節省多少人工時間？

實驗三：跨工站泛化性比較
  └─ 對新工站的適應能力是否優於有監督的 YOLO？

實驗四：下游模型品質驗證
  └─ 用輔助標注的數據訓練的 YOLO，效果有沒有明顯下降？
```

### 1.3 工具與角色定位

| 工具 | 角色 | 說明 |
|---|---|---|
| GroundingDINO | 標注輔助工具 | 自動生成初始邊界框，減少人工從零標注 |
| 人工審核員 | 品質把關 | 審核低信心框、補標漏偵測 |
| YOLO | 最終部署模型 | 用審核後的標注數據訓練，實際部署推論 |

> **重要**：GroundingDINO 與 YOLO 並非競爭關係。GroundingDINO 是生產數據的工具，YOLO 是消費數據的模型。

---

## 2. 資料集規劃與切分策略

### 2.1 資料概況確認

執行實驗前，先整理各工站的現有資料狀況：

| 欄位 | 說明 | 範例 |
|---|---|---|
| 工站 ID | 唯一識別碼 | WS-01 ~ WS-10 |
| 工站類型 | 焊接 / 檢測 | 焊接 |
| 已標注圖像數 | 現有 ground truth 數量 | 150 張 |
| 標注類別 | 偵測目標的名稱 | 焊點、焊道 |
| 特殊條件 | 曝光、視野、角度等差異 | 背光、廣角 |

### 2.2 切分原則

**核心原則：以「工站」為單位切分，而非以「圖像」為單位。**

錯誤做法（不要這樣做）：
```
❌ 把所有工站圖像混合後隨機切 80% train / 20% test
   → 測試集可能包含訓練集工站的圖像，無法評估真正的泛化能力
```

正確做法：
```
✅ 保留 2~3 個完整工站作為測試集，其餘工站的數據用於訓練
   → 測試集代表「模型從未見過的新工站」，評估結果有實際意義
```

### 2.3 建議的資料切分方案

假設共有 10 個工站（8 個焊接工站 + 2 個檢測工站）：

```
Fine-tune 集（訓練用）：
  WS-01 ~ WS-07（7 個焊接工站）
  各站取全部標注數據的 80% 作為訓練

Fine-tune 驗證集：
  WS-01 ~ WS-07 各站剩餘 20%
  用於 fine-tune 過程中監控是否 overfitting

泛化測試集（完全保留，不參與任何訓練）：
  WS-08（焊接工站，條件與訓練集差異最大的）
  WS-09、WS-10（檢測工站，若要單獨評估）
```

> **選擇測試工站的依據**：優先選擇拍攝條件（曝光、角度、視野）與訓練集差異較大的工站，以評估最具挑戰性的泛化場景。

### 2.4 各工站標注數據分配表（範本）

| 工站 ID | 類型 | 總標注數 | Fine-tune 訓練 | Fine-tune 驗證 | 測試集 |
|---|---|---|---|---|---|
| WS-01 | 焊接 | 180 | 144 | 36 | - |
| WS-02 | 焊接 | 150 | 120 | 30 | - |
| WS-03 | 焊接 | 200 | 160 | 40 | - |
| WS-04 | 焊接 | 160 | 128 | 32 | - |
| WS-05 | 焊接 | 170 | 136 | 34 | - |
| WS-06 | 焊接 | 190 | 152 | 38 | - |
| WS-07 | 焊接 | 140 | 112 | 28 | - |
| WS-08 | 焊接 | 150 | - | - | 150 |
| WS-09 | 檢測 | 200 | - | - | 200 |
| WS-10 | 檢測 | 180 | - | - | 180 |

### 2.5 資料夾結構建議

```
dataset/
├── finetune/
│   ├── train/
│   │   ├── images/
│   │   │   ├── WS01_001.jpg
│   │   │   └── ...
│   │   └── labels/          # COCO JSON 或 YOLO txt 格式
│   │       ├── WS01_001.txt
│   │       └── ...
│   └── val/
│       ├── images/
│       └── labels/
└── test/
    ├── WS08/
    │   ├── images/
    │   └── labels/           # ground truth，僅用於評估，不參與訓練
    ├── WS09/
    └── WS10/
```

### 2.6 標注格式統一

確保所有標注數據統一格式後再進行實驗。建議使用 **COCO JSON 格式**作為中間格式，以便 GroundingDINO 和 YOLO 各自轉換使用。

```json
{
  "images": [
    {"id": 1, "file_name": "WS01_001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "weld_point"},
    {"id": 2, "name": "weld_bead"}
  ]
}
```

---

## 3. GroundingDINO Fine-tune 流程

### 3.1 為什麼需要 Fine-tune

GroundingDINO 預訓練資料以自然圖像為主（COCO、Objects365 等），工業焊接場景具有以下特殊性：

- 焊點外觀與日常語言描述差距大（高反光、不規則形狀）
- 治具、夾具等工業背景元素干擾
- 圖像構圖高度一致（固定機位拍攝）

Zero-shot 在此場景表現不穩定，**建議使用現有標注數據進行 fine-tune**。

### 3.2 Text Prompt 設計

Fine-tune 前先確定 text prompt，建議：

```
主要類別：
  "weld point"（或 "焊點"，視訓練語言而定）
  "weld bead"（若需偵測焊道）

檢測工站額外類別：
  "small component"（第一階段：偵測小件）
  "solder joint"（第二階段：在 crop 圖中偵測焊點）
```

> **注意**：prompt 用詞需要在 fine-tune 訓練和推論時保持一致，不可隨意更換。

### 3.3 Fine-tune 超參數建議

| 超參數 | 建議值 | 說明 |
|---|---|---|
| Backbone | Swin-T 或 Swin-B | 資料量少用 Swin-T，有 1000+ 張用 Swin-B |
| Learning Rate | 1e-4 ~ 5e-5 | 從低學習率開始 |
| Batch Size | 2 ~ 4 | 受 GPU 記憶體限制 |
| Epochs | 30 ~ 50 | 配合 Early Stopping |
| Early Stopping | Patience = 5 | 監控驗證集 AP50 |
| Image Size | 800 x 1333 | GroundingDINO 預設輸入 |
| Data Augmentation | 水平翻轉、亮度/對比調整 | 模擬不同廠區拍攝條件 |

### 3.4 Fine-tune 流程圖

```
準備 Fine-tune 訓練集（WS-01 ~ WS-07 各站 80%）
            ↓
設定 Text Prompt（"weld point" 等）
            ↓
開始訓練（監控驗證集 AP50）
            ↓
Early Stopping（驗證集 AP50 不再提升）
            ↓
儲存最佳模型權重
            ↓
在測試集（WS-08 ~ WS-10）上評估（實驗一、三）
```

### 3.5 Confidence 閾值設定

Fine-tune 完成後，需要決定「自動接受」的信心門檻：

```
推論輸出：每個框帶有 confidence score（0~1）

建議策略：
  confidence ≥ 0.7  → 自動接受，直接採用
  0.3 ≤ confidence < 0.7  → 標記為「需人工審核」
  confidence < 0.3  → 忽略（或列入人工重新標注）
```

> 閾值應根據實驗一的 Precision-Recall 曲線動態調整，**不建議固定一個值直接使用**，應以「可接受的 Recall 損失換取最高自動接受率」為原則。

---

## 4. 實驗一：標注品質驗證

**目的**：量化 GroundingDINO fine-tune 後的標注品質，與人工 ground truth 比較。

**回答的問題**：「它標得出來嗎？標得準嗎？」

### 4.1 實驗設計

```
輸入：測試集圖像（WS-08 ~ WS-10，共 N 張）
模型：Fine-tune 後的 GroundingDINO
輸出：預測邊界框（含 confidence score）
比較基準：人工標注的 ground truth
```

### 4.2 主要評估指標

#### 4.2.1 Recall（召回率）

$$\text{Recall} = \frac{TP}{TP + FN}$$

- **TP（True Positive）**：預測框與 ground truth 框的 IoU ≥ 閾值
- **FN（False Negative）**：ground truth 框沒有對應的預測框（漏偵測）
- **重要性**：標注場景中 **Recall 優先**，漏標比誤標更嚴重

#### 4.2.2 Precision（精確率）

$$\text{Precision} = \frac{TP}{TP + FP}$$

- **FP（False Positive）**：預測框與任何 ground truth 框的 IoU < 閾值（誤偵測）
- **重要性**：影響人工審核工作量，Precision 越高，需要刪除的錯誤框越少

#### 4.2.3 F1-Score

$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

用於在 Precision 和 Recall 之間取平衡，作為單一綜合指標報告。

#### 4.2.4 AP50 / AP75 / mAP

| 指標 | IoU 閾值 | 說明 |
|---|---|---|
| AP50 | 0.5 | 標注輔助場景的主要指標，框的位置大致對即可 |
| AP75 | 0.75 | 對框的精確度要求更高 |
| mAP | 0.5:0.05:0.95 | COCO 標準，綜合評估 |

#### 4.2.5 自動接受率

$$\text{自動接受率} = \frac{\text{confidence} \geq T \text{ 的框數}}{\text{所有預測框數}}$$

此指標直接反映「多少比例的框不需要人工處理」。

### 4.3 結果呈現格式

#### 各工站標注品質彙整表

| 工站 | Recall@IoU0.5 | Precision@IoU0.5 | F1 | AP50 | 自動接受率(T=0.7) |
|---|---|---|---|---|---|
| WS-08 | 0.87 | 0.81 | 0.84 | 0.79 | 73% |
| WS-09 | 0.82 | 0.78 | 0.80 | 0.75 | 68% |
| WS-10 | 0.85 | 0.83 | 0.84 | 0.77 | 71% |
| **平均** | **0.85** | **0.81** | **0.83** | **0.77** | **71%** |

#### Precision-Recall 曲線

針對每個測試工站畫出 PR 曲線，並標注不同 confidence 閾值的操作點：

```
Precision
  1.0 |
  0.9 |        ●  ← T=0.9（高精確，低召回）
  0.8 |      ●
  0.7 |    ●       ← T=0.7（建議操作點）
  0.6 |  ●
  0.5 | ●           ← T=0.3（高召回，低精確）
      +─────────────────── Recall
        0.6  0.7  0.8  0.9  1.0
```

#### 誤差分析：錯誤案例分類

統計錯誤類型，有助於後續改進：

| 錯誤類型 | 數量 | 比例 | 主要原因 |
|---|---|---|---|
| 漏偵測（FN） | X | X% | 焊點被治具遮擋 / 曝光過度 |
| 誤偵測（FP） | X | X% | 治具開孔被誤判為焊點 |
| 框偏移（Low IoU）| X | X% | 焊點邊界不清晰 |

### 4.4 通過標準建議

| 指標 | 最低通過標準 | 建議目標 |
|---|---|---|
| Recall@IoU0.5 | ≥ 0.80 | ≥ 0.85 |
| Precision@IoU0.5 | ≥ 0.70 | ≥ 0.80 |
| AP50 | ≥ 0.70 | ≥ 0.75 |
| 自動接受率 | ≥ 60% | ≥ 70% |

> Recall 最低通過標準高於 Precision，因為漏標會直接影響下游模型訓練品質。

---

## 5. 實驗二：標注效率比較

**目的**：量化導入 GroundingDINO 後實際節省的人工時間。

**回答的問題**：「導入後能省多少工？」

### 5.1 實驗設計

選取一個測試工站（建議 WS-08）的全部圖像（約 150 張）進行對比：

```
流程 A（傳統純人工）：
  標注員從零對 150 張圖進行人工標注
  全程計時，記錄總時間 T_manual

流程 B（GroundingDINO 輔助）：
  Step 1：GroundingDINO 自動推論所有圖像（計時 T_inference）
  Step 2：標注員審核自動框
    - confidence ≥ 0.7：快速確認或刪除（計時 T_review）
    - confidence < 0.7：決定接受/修改/刪除
  Step 3：補標漏偵測的物件（計時 T_add）

T_new = T_inference + T_review + T_add
```

> **人員要求**：兩個流程應由熟練程度相近的標注員執行，或同一人執行兩次（間隔足夠時間避免記憶效應）。

### 5.2 記錄表格

#### 流程 A：純人工標注記錄

| 圖像批次 | 圖像數 | 開始時間 | 結束時間 | 耗時（分鐘）| 每張平均耗時 |
|---|---|---|---|---|---|
| Batch 1 | 50 | - | - | - | - |
| Batch 2 | 50 | - | - | - | - |
| Batch 3 | 50 | - | - | - | - |
| **合計** | **150** | - | - | **T_manual** | - |

#### 流程 B：GroundingDINO 輔助記錄

| 階段 | 說明 | 圖像數 | 耗時（分鐘）|
|---|---|---|---|
| 自動推論 | GroundingDINO 批次推論 | 150 | T_inference |
| 審核-接受 | confidence ≥ 0.7，快速確認 | X 張 | T_review_accept |
| 審核-修改 | 需調整邊界框 | X 張 | T_review_modify |
| 審核-刪除 | 誤偵測，刪除錯誤框 | X 框 | T_review_delete |
| 人工補標 | 完全漏偵測，從零標注 | X 張 | T_add |
| **合計** | - | **150** | **T_new** |

### 5.3 輸出指標

| 指標 | 計算方式 | 目標 |
|---|---|---|
| **時間節省率** | `(T_manual - T_new) / T_manual × 100%` | ≥ 50% |
| **自動接受率** | `無需修改的框數 / 總預測框數 × 100%` | ≥ 70% |
| **純人工補標率** | `需從零標注的圖像數 / 總圖像數 × 100%` | ≤ 15% |
| **每張圖平均審核時間** | `T_new / 150` | < `T_manual / 150 × 50%` |

### 5.4 結果呈現（報告用圖表）

建議用以下圖表呈現給主管：

**圖表一：時間對比長條圖**
```
傳統流程   ████████████████████████  300 分鐘 (100%)
GroundingDINO 輔助  ██████████  120 分鐘 (40%)
                                   節省 60%
```

**圖表二：GroundingDINO 輔助流程時間分解**
```
自動推論    ██  10 分鐘 (8%)
快速審核    ██████  55 分鐘 (46%)
人工補標    █████  55 分鐘 (46%)
```

**圖表三：人工介入類型分布（圓餅圖）**
```
直接接受  71%
需修改    12%
需刪除     8%
需補標     9%
```

---

## 6. 實驗三：跨工站泛化性比較

**目的**：證明 GroundingDINO fine-tune 後對新工站的適應能力優於等量數據訓練的 YOLO。

**回答的問題**：「為什麼不直接用 YOLO 輔助標注？」

### 6.1 實驗設計

```
兩個模型使用完全相同的訓練數據：
  - Fine-tune GroundingDINO（WS-01 ~ WS-07 數據）
  - 訓練 YOLO（WS-01 ~ WS-07 相同數據）

在相同測試集上評估：
  - WS-08（焊接，拍攝條件差異較大）
  - WS-09、WS-10（檢測工站）
```

> **此實驗的關鍵**：控制訓練數據量完全相同，確保差異只來自模型架構和預訓練數據。

### 6.2 評估指標

在每個測試工站上，同時計算兩個模型的：

- Recall@IoU0.5
- Precision@IoU0.5
- AP50
- 特別關注在「拍攝條件差異大」的工站上的指標差距

### 6.3 結果呈現格式

#### 跨工站泛化性對比表

| 測試工站 | 指標 | Fine-tune GroundingDINO | YOLO（相同訓練數據）| 差異 |
|---|---|---|---|---|
| WS-08（焊接）| Recall | 0.85 | 0.71 | +0.14 |
| WS-08（焊接）| AP50 | 0.77 | 0.63 | +0.14 |
| WS-09（檢測）| Recall | 0.82 | 0.65 | +0.17 |
| WS-09（檢測）| AP50 | 0.74 | 0.59 | +0.15 |

#### 各工站 AP50 折線圖

```
AP50
1.0 |
0.9 |
0.8 |  ●──●──●──●──●──●──●     ← GroundingDINO（訓練站）
0.7 |  ○──○──○──○──○──○──○     ← YOLO（訓練站）
    |                    ↓ 新工站
0.6 |                    ●       ← GroundingDINO（測試站）
0.5 |                    ○       ← YOLO（測試站）
    +────────────────────────── 工站
      WS01 WS02 WS03 WS04 WS05 WS06 WS07 WS08
```

**預期結論**：兩個模型在訓練工站上表現相近，但在未見工站（WS-08~WS-10）上，GroundingDINO 的 Recall 下降幅度明顯小於 YOLO，代表其泛化能力更強，更適合作為新工站的標注輔助工具。

### 6.4 補充說明：冷啟動場景

除了上述等量數據比較，可以額外呈現「零標注數據」場景：

| 情境 | GroundingDINO | YOLO |
|---|---|---|
| 0 張標注數據（全新工站） | Zero-shot 可用（Recall ~0.6） | 無法使用 |
| 50 張標注數據 | Fine-tune 後 Recall ~0.80 | 可訓練但泛化差 |
| 150 張標注數據 | Fine-tune 後 Recall ~0.85 | 效果接近（但局限於訓練分布）|

此表格直觀說明：**GroundingDINO 在冷啟動和少量數據場景下的優勢最為明顯**。

---

## 7. 實驗四：下游模型品質驗證

**目的**：證明使用 GroundingDINO 輔助標注的數據訓練出的 YOLO，效果與使用人工 ground truth 訓練的 YOLO 相當。

**回答的問題**：「用你的方法，最後的模型還能用嗎？」

### 7.1 實驗設計

以 WS-08 為目標工站（測試泛化場景）：

```
YOLO_A（基準）：
  訓練數據：WS-08 的人工 ground truth（150 張）
  → 評估 YOLO_A 在 WS-08 測試集的效果

YOLO_B（輔助標注方案）：
  訓練數據：WS-08 的 GroundingDINO 輔助標注 + 人工審核後的標注（150 張）
  → 評估 YOLO_B 在 WS-08 測試集的效果

測試集：從 WS-08 中預先保留的 30 張圖像（兩個模型都不看到）
```

> **操作細節**：
> - WS-08 共 150 張，取 120 張作為訓練（分別用人工標注 vs GroundingDINO 輔助標注），保留 30 張作為共同測試集。
> - 30 張測試集的標注一律使用人工 ground truth，確保評估基準一致。

### 7.2 評估指標

在共同測試集上，比較 YOLO_A 和 YOLO_B 的：

| 指標 | YOLO_A（人工標注）| YOLO_B（輔助標注）| 差異 | 可接受範圍 |
|---|---|---|---|---|
| Recall@IoU0.5 | - | - | - | ≤ 5% 下降 |
| Precision@IoU0.5 | - | - | - | ≤ 5% 下降 |
| AP50 | - | - | - | ≤ 5% 下降 |
| mAP | - | - | - | ≤ 5% 下降 |

### 7.3 標注品質對下游模型的影響分析

可以進一步分析不同「人工審核比例」對 YOLO_B 的影響：

| 場景 | 人工審核比例 | YOLO_B AP50 | 說明 |
|---|---|---|---|
| 全人工（基準） | 100% | X.XX | YOLO_A |
| 嚴格審核 | 100%（審核但使用輔助標注） | X.XX | YOLO_B-Strict |
| 標準審核 | confidence < 0.7 才審核 | X.XX | YOLO_B-Standard |
| 輕度審核 | confidence < 0.5 才審核 | X.XX | YOLO_B-Light |
| 無審核 | 0%（直接使用 GroundingDINO 輸出）| X.XX | YOLO_B-Auto |

此分析可協助決策者根據可接受的品質損失，選擇對應的審核策略。

### 7.4 結果解讀

- 若 YOLO_B ≈ YOLO_A（差異 ≤ 5%）：**方案成立**，輔助標注品質足夠
- 若 YOLO_B < YOLO_A（差異 > 5%）：分析差距來源，針對性補強（增加人工審核比例或調低 confidence 閾值）

---

## 8. 驗證報告撰寫結構

### 8.1 報告大綱

```
1. 執行摘要（1 頁）
   - 背景問題
   - 核心結論
   - 效益量化（節省 X% 時間，模型效果差異 ≤ Y%）

2. 背景與問題定義（1~2 頁）
   - 現有標注流程的瓶頸
   - 各工站數據現況
   - 本次評估目標

3. 解決方案概述（1 頁）
   - GroundingDINO 介紹與角色定位
   - 整體流程說明
   - 工具選型理由

4. 實驗設計（2~3 頁）
   - 資料集切分策略
   - Fine-tune 設定
   - 各實驗說明

5. 實驗結果（3~4 頁）
   5.1 標注品質驗證（實驗一）
   5.2 標注效率比較（實驗二）
   5.3 跨工站泛化性（實驗三）
   5.4 下游模型品質（實驗四）

6. 結論與建議（1 頁）
   - 達成的效益
   - 建議的導入策略
   - 後續改善方向

7. 附錄
   - 詳細指標數據表
   - 標注工具設定截圖
   - 失敗案例分析
```

### 8.2 核心頁面：執行摘要模板

```
【背景】
  當前每個新工站需要 X 小時人工標注 N 張圖像，
  人力成本高且難以快速擴展至新廠區。

【方案】
  導入 GroundingDINO 作為標注輔助工具，
  自動生成初始邊界框，人工僅需審核低信心框。

【結果】
  ✅ 標注時間節省：從 X 小時降至 Y 小時（節省 Z%）
  ✅ 自動接受率：XX% 的框無需人工干預
  ✅ 下游模型品質：AP50 差異 ≤ 5%（YOLO_A vs YOLO_B）
  ✅ 跨工站泛化：新工站 Recall 高出傳統 YOLO XX 個百分點

【建議】
  建議正式導入 GroundingDINO 輔助標注流程，
  預計可將新工站上線週期從 X 天縮短至 Y 天。
```

---

## 9. 常見問題與主管答辯準備

### Q1：為什麼不直接用 YOLO 來輔助標注？

**回答**：

YOLO 輔助標注（pseudo-labeling）在業界確實存在，但有一個根本限制：

> 需要先有標注數據才能訓練，無法解決「第一批數據如何標注」的問題。

GroundingDINO 的優勢在三個場景：

1. **冷啟動**：全新工站沒有任何標注數據時，GroundingDINO 可以直接從文字描述開始輔助標注；YOLO 這時候根本無法使用。
2. **跨工站泛化**：如實驗三所示，在拍攝條件有差異的新工站上，GroundingDINO 的 Recall 下降幅度明顯小於 YOLO。
3. **類別擴充**：若需要新增偵測類別（例如從焊點增加到焊道），GroundingDINO 只需修改文字描述，YOLO 需要重新標注並訓練。

---

### Q2：GroundingDINO 推論很慢，這樣有節省時間嗎？

**回答**：

GroundingDINO 的推論速度確實比 YOLO 慢（約 5~10 倍），但在標注輔助場景中，這個差距影響有限：

| 項目 | 時間估算 |
|---|---|
| GroundingDINO 推論 150 張圖 | ~10 分鐘（批次處理，離線運行）|
| 節省的人工標注時間 | ~180 分鐘（估計節省 60%）|

推論可以在非工作時間批次執行，人工只需在第二天審核結果，整體週期並不增加。

**最終部署的模型仍然是 YOLO**，推論速度不受影響。

---

### Q3：模型在沒見過的工站表現不好怎麼辦？

**回答**：

這是預期內的情況，有標準的處理流程：

```
Step 1：先用 GroundingDINO 零樣本標注（Recall ~0.6~0.7）
Step 2：人工補標漏偵測的部分
Step 3：累積 50~100 張標注後，做一次針對新工站的 fine-tune
Step 4：重新評估，Recall 應提升至 0.80 以上
```

這個過程本質上是「逐步降低人工依賴」，第一批仍需較多人工，但後續效率逐漸提升。

---

### Q4：Fine-tune GroundingDINO 需要多少計算資源？

**回答**：

| 資源 | 最低需求 | 建議配置 |
|---|---|---|
| GPU | NVIDIA RTX 3090（24GB）| A100 40GB |
| 訓練時間（1000 張，30 epochs）| ~2 小時 | ~45 分鐘 |
| 儲存空間 | 模型權重約 700MB | - |

Fine-tune 為一次性投資，完成後模型可重複使用，無需頻繁重新訓練。

---

### Q5：標注一致性如何保證？

**回答**：

人工標注存在標注員之間的不一致性（Inter-annotator Variance），GroundingDINO 輔助標注反而有助於提升一致性：

- 自動生成的初始框作為「錨點」，減少標注員的主觀判斷
- 一致的 confidence 閾值作為接受標準，減少個人差異
- 可追蹤哪些框是自動生成、哪些是人工修改，便於後續品質稽核

---

## 10. 附錄：指標計算說明

### A. IoU（Intersection over Union）計算

$$\text{IoU} = \frac{\text{預測框} \cap \text{Ground Truth 框}}{\text{預測框} \cup \text{Ground Truth 框}}$$

```python
def calculate_iou(box_pred, box_gt):
    """
    box format: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box_pred[0], box_gt[0])
    y1 = max(box_pred[1], box_gt[1])
    x2 = min(box_pred[2], box_gt[2])
    y2 = min(box_pred[3], box_gt[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_pred = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    area_gt = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    union = area_pred + area_gt - intersection

    return intersection / union if union > 0 else 0
```

### B. TP / FP / FN 判斷邏輯

```python
def match_predictions_to_gt(predictions, ground_truths, iou_threshold=0.5):
    """
    predictions: list of [x1, y1, x2, y2, confidence]
    ground_truths: list of [x1, y1, x2, y2]
    """
    # 依 confidence 由高到低排序
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    matched_gt = set()
    tp, fp = 0, 0

    for pred in predictions:
        best_iou = 0
        best_gt_idx = -1
        for idx, gt in enumerate(ground_truths):
            if idx in matched_gt:
                continue
            iou = calculate_iou(pred[:4], gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truths) - len(matched_gt)
    return tp, fp, fn
```

### C. 評估腳本架構建議

```python
# evaluate.py 主要架構

import json
from pathlib import Path

def evaluate_model(model, test_dataset_path, iou_threshold=0.5, conf_threshold=0.7):
    results = {}

    for station_dir in Path(test_dataset_path).iterdir():
        station_id = station_dir.name
        tp_total, fp_total, fn_total = 0, 0, 0
        auto_accept_count, total_pred_count = 0, 0

        for image_path in station_dir.glob("images/*.jpg"):
            gt_path = image_path.parent.parent / "labels" / image_path.with_suffix(".json").name
            ground_truths = load_ground_truth(gt_path)
            predictions = model.predict(image_path)

            # 統計自動接受率
            total_pred_count += len(predictions)
            auto_accept_count += sum(1 for p in predictions if p[4] >= conf_threshold)

            # 過濾低信心框（模擬實際標注流程）
            filtered_predictions = [p for p in predictions if p[4] >= conf_threshold]

            tp, fp, fn = match_predictions_to_gt(filtered_predictions, ground_truths, iou_threshold)
            tp_total += tp
            fp_total += fp
            fn_total += fn

        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        auto_accept_rate = auto_accept_count / total_pred_count if total_pred_count > 0 else 0

        results[station_id] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "auto_accept_rate": round(auto_accept_rate, 4),
            "tp": tp_total, "fp": fp_total, "fn": fn_total
        }

    return results
```

### D. 推薦工具清單

| 工具 | 用途 | 備註 |
|---|---|---|
| [Label Studio](https://labelstud.io/) | 標注工具 + 與 ML 模型整合 | 支援 GroundingDINO 作為 backend |
| [CVAT](https://www.cvat.ai/) | 標注工具 | 支援半自動標注 |
| [supervision](https://github.com/roboflow/supervision) | 偵測結果後處理、視覺化 | Roboflow 開發 |
| [pycocotools](https://github.com/cocodataset/cocoapi) | COCO 格式 AP 計算 | 標準評估工具 |
| [Weights & Biases](https://wandb.ai/) | 實驗追蹤 | Fine-tune 過程記錄 |

---

*文件版本：v1.0*
*最後更新：2026-03*
*適用工站類型：焊接工站、檢測工站*
