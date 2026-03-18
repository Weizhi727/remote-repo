# 機台調整圖像模擬腳本使用指南

> **適用場景**：在進行 YOLO / GroundingDINO 泛化性評估時，模擬量產前頻繁機台調整所造成的圖像變化，取代實際重拍的成本。

---

## 目錄

1. [為什麼需要圖像模擬？](#1-為什麼需要圖像模擬)
2. [環境安裝](#2-環境安裝)
3. [腳本架構總覽](#3-腳本架構總覽)
4. [八種模擬效果詳解](#4-八種模擬效果詳解)
5. [快速上手：三種使用情境](#5-快速上手三種使用情境)
6. [輸出資料結構](#6-輸出資料結構)
7. [搭配泛化性評估的完整流程](#7-搭配泛化性評估的完整流程)
8. [參數調整指南](#8-參數調整指南)
9. [常見問題](#9-常見問題)

---

## 1. 為什麼需要圖像模擬？

### 1.1 問題背景

在工廠量產前，工程師通常需要反覆調整機台，例如：

- 治具重新定位 → 圖像平移、旋轉
- 重新對焦或鎖定鏡頭 → 模糊、視野縮放改變
- 更換燈管 → 曝光、色溫改變
- 傳送帶震動 → 運動模糊

每次調整後，**攝影機拍出的圖像與原始訓練數據有肉眼可見的差異**，這會直接影響 YOLO / GroundingDINO 的偵測結果。

### 1.2 傳統做法的問題

| 做法 | 問題 |
|---|---|
| 實際調整機台重拍 | 需要佔用生產線，成本高 |
| 直接用未見工站數據測試 | 只能評估「整體泛化性」，無法定位「哪種變化最脆弱」|
| 不做測試直接部署 | 部署後才發現 Recall 下降，補救成本更高 |

### 1.3 圖像模擬的優勢

```
原始測試集圖像
      ↓ simulate_machine_adjustment.py（一次執行）
15 種單項模擬圖像 + N 種複合模擬圖像
      ↓
精確定位模型弱點 → 針對性補強（增加對應數據增強或重標數據）
```

**優點**：
- 不需要佔用生產線
- 可控制每種變化的強度
- 結果可重現（固定 seed）
- 可比較 YOLO 和 GroundingDINO 對各類變化的抵抗力差異

---

## 2. 環境安裝

### 2.1 依賴套件

```bash
pip install opencv-python numpy
```

| 套件 | 版本需求 | 用途 |
|---|---|---|
| `opencv-python` | ≥ 4.5 | 圖像讀寫、幾何變換、濾波 |
| `numpy` | ≥ 1.21 | 數值運算、矩陣操作 |

### 2.2 確認安裝成功

```bash
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
```

### 2.3 支援的圖像格式

| 格式 | 副檔名 |
|---|---|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| BMP | `.bmp` |
| TIFF | `.tiff` |

---

## 3. 腳本架構總覽

```
simulate_machine_adjustment.py
│
├── 八種單項模擬函數
│   ├── simulate_roi_blur()        # 對焦模糊
│   ├── simulate_motion_blur()     # 運動模糊
│   ├── simulate_exposure()        # 曝光變化
│   ├── simulate_contrast()        # 對比度變化
│   ├── simulate_fov_change()      # 視野變化（縮放/平移/旋轉）
│   ├── simulate_white_balance()   # 色溫偏移
│   ├── simulate_vignette()        # 鏡頭暗角
│   └── simulate_noise()           # 感光雜訊
│
├── 複合擾動
│   └── apply_compound_perturbation()  # 隨機組合多種擾動
│
├── 批次處理
│   ├── run_single_perturbations()     # 批次產生 15 種單項樣本
│   └── run_compound_perturbations()   # 批次產生複合樣本 + JSON 記錄
│
└── main()  # 命令列入口
```

### 命令列參數一覽

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--input` / `-i` | 必填 | 輸入圖像或資料夾路徑 |
| `--output` / `-o` | `output_simulated` | 輸出資料夾 |
| `--mode` | `all` | `all` / `single` / `compound` |
| `--n_compounds` | `5` | 每張圖片產生幾種複合擾動 |
| `--n_perturbations` | `3` | 每次複合擾動套用幾種效果 |
| `--seed` | `42` | 隨機種子（確保可重現） |

---

## 4. 八種模擬效果詳解

### 4.1 ROI 模糊（`simulate_roi_blur`）

**模擬情境**：對焦環鬆動、鏡頭表面污染、重新裝配鏡頭後對焦偏移

**技術實現**：Gaussian Blur，kernel size 和 sigma 隨 intensity 等比例增大

| `intensity` 值 | 視覺效果 | 對應情境 |
|---|---|---|
| 0.0 ~ 0.3 | 輕微模糊，邊緣稍糊 | 輕微對焦偏移 |
| 0.3 ~ 0.7 | 中度模糊，肉眼可見 | 明顯對焦偏移 |
| 0.7 ~ 1.0 | 嚴重模糊，焊點輪廓不清 | 鏡頭嚴重失焦 |

```python
# 直接呼叫範例
from simulate_machine_adjustment import simulate_roi_blur
import cv2

image = cv2.imread("WS08_001.jpg")
blurred = simulate_roi_blur(image, intensity=0.5)
cv2.imwrite("blurred.jpg", blurred)
```

---

### 4.2 運動模糊（`simulate_motion_blur`）

**模擬情境**：傳送帶震動、機械手臂晃動、相機快門速度設定過慢

**技術實現**：方向性 kernel（1D 線條），套用指定角度的仿射旋轉

| 參數 | 說明 | 建議範圍 |
|---|---|---|
| `intensity` | 模糊長度（越大越長） | 0.1 ~ 0.5 |
| `angle` | 模糊方向（度）| 0（水平）/ 90（垂直）|

---

### 4.3 曝光變化（`simulate_exposure`）

**模擬情境**：光源老化亮度下降、更換不同瓦數燈管、調整相機 ISO

**技術實現**：圖像像素乘以 `2^ev_shift`

| `ev_shift` 值 | 視覺效果 | 對應情境 |
|---|---|---|
| `+1.5` | 明顯過曝，亮部細節消失 | 換高亮度燈管 |
| `+0.5` | 輕微過亮 | 環境光增強 |
| `-0.5` | 輕微過暗 | 燈管老化 |
| `-1.5` | 明顯欠曝，暗部無細節 | 燈管快熄滅 |

> **注意**：過曝和欠曝都會嚴重影響焊點反光特徵，是最常見的模型失效原因之一。

---

### 4.4 對比度變化（`simulate_contrast`）

**模擬情境**：反光板污染、光源角度偏移、背景材質更換

**技術實現**：線性變換 `pixel = alpha × pixel + beta`

| 參數 | 效果 | 對應情境 |
|---|---|---|
| `alpha < 1`（如 0.6）| 對比壓縮，圖像偏灰 | 反光板污染、均勻散射光 |
| `alpha > 1`（如 1.6）| 對比拉伸，亮暗更極端 | 直射光、強聚焦光源 |
| `beta > 0` | 整體偏亮 | 搭配 alpha 模擬過曝+低對比 |
| `beta < 0` | 整體偏暗 | 搭配 alpha 模擬欠曝+低對比 |

---

### 4.5 視野變化（`simulate_fov_change`）

**模擬情境**：治具重新定位、鏡頭重新鎖定螺絲後位移、更換不同焦距鏡頭

**技術實現**：仿射變換（Affine Transform），結合縮放、平移、旋轉

| 參數 | 效果 | 建議範圍 |
|---|---|---|
| `zoom > 1`（如 1.15）| 圖像放大，視野縮小 | 拉近鏡頭 |
| `zoom < 1`（如 0.85）| 圖像縮小，視野擴大 | 拉遠鏡頭 |
| `tx`（如 0.08）| 水平平移 8% 圖寬 | 治具左右偏移 |
| `ty`（如 0.05）| 垂直平移 5% 圖高 | 治具前後偏移 |
| `rotation`（如 3.0）| 旋轉 3 度 | 鏡頭鎖定角度偏差 |

> **邊界填充**：採用 `BORDER_REPLICATE`（邊緣像素延伸），避免平移後出現明顯黑邊。

---

### 4.6 色溫偏移（`simulate_white_balance`）

**模擬情境**：更換不同色溫燈管（例如從 4000K 換成 6500K）、白平衡設定跑掉

**技術實現**：分別調整 BGR 三個 channel 的增益

| 設定 | 視覺效果 | 對應情境 |
|---|---|---|
| `r_shift=1.2, b_shift=0.85` | 偏暖橙 | 換暖白燈管 |
| `r_shift=0.85, b_shift=1.2` | 偏冷藍 | 換冷白燈管 / 日光燈 |
| `r_shift=1.0, b_shift=1.0` | 無變化（基準）| - |

---

### 4.7 鏡頭暗角（`simulate_vignette`）

**模擬情境**：鏡頭安裝偏心、加裝遮光罩位置偏移

**技術實現**：以圖像中心為基準，距離越遠亮度越低的放射狀 mask

| `strength` 值 | 效果 |
|---|---|
| 0.2 | 輕微暗角，僅四角稍暗 |
| 0.5 | 中度暗角，邊緣明顯偏暗 |
| 0.7 | 強烈暗角，僅中央亮 |

---

### 4.8 感光雜訊（`simulate_noise`）

**模擬情境**：換不同型號攝影機、提高 ISO 值、溫度過高導致感光元件雜訊增加

**技術實現**：加入 Gaussian 隨機雜訊（零均值，標準差 `std`）

| `std` 值 | 效果 |
|---|---|
| 5 | 輕微顆粒感，幾乎不影響辨識 |
| 15 | 中度雜訊，細節稍微模糊 |
| 25 | 明顯雜訊，低對比細節消失 |

---

## 5. 快速上手：三種使用情境

### 情境一：快速預覽單張圖片的所有效果

```bash
python simulate_machine_adjustment.py \
  --input images/WS08_001.jpg \
  --output output_preview/ \
  --mode all
```

**輸出**：`output_preview/` 下會有 15 個子資料夾（各單項效果）和 1 個 `compound/` 資料夾（5 種複合效果）。

---

### 情境二：批次處理整個測試集，只產生複合擾動

適合直接拿來評估模型泛化性：

```bash
python simulate_machine_adjustment.py \
  --input dataset/test/WS08/images/ \
  --output output_generalization/ \
  --mode compound \
  --n_compounds 10 \
  --n_perturbations 3 \
  --seed 42
```

**說明**：
- `--n_compounds 10`：每張圖產生 10 種不同的複合擾動組合
- `--n_perturbations 3`：每次擾動從 8 種效果中隨機選 3 種套用
- `--seed 42`：固定隨機種子，確保他人可重現相同結果

---

### 情境三：精準測試特定效果（修改 `PERTURBATION_RANGES`）

若只想測試「曝光」和「視野」兩種變化：

```python
# 在腳本中修改 PERTURBATION_RANGES（或複製到你自己的腳本中）
PERTURBATION_RANGES = {
    "roi_blur":      {"enabled": False, ...},  # 關閉
    "motion_blur":   {"enabled": False, ...},  # 關閉
    "exposure":      {"enabled": True, "ev_shift": (-2.0, 2.0)},  # 加大範圍
    "contrast":      {"enabled": False, ...},
    "fov":           {"enabled": True, "zoom": (0.8, 1.2), "tx": (-0.1, 0.1), ...},
    "white_balance": {"enabled": False, ...},
    "vignette":      {"enabled": False, ...},
    "noise":         {"enabled": False, ...},
}
```

然後執行：

```bash
python simulate_machine_adjustment.py \
  --input images/ --output output_exp_fov/ \
  --mode compound --n_compounds 20
```

---

## 6. 輸出資料結構

### 6.1 `--mode all` 或 `--mode single` 的輸出

```
output_simulated/
├── blur_light/            # 單項：輕度 ROI 模糊（intensity=0.2）
│   ├── WS08_001_blur_light.jpg
│   └── ...
├── blur_heavy/            # 單項：重度 ROI 模糊（intensity=0.7）
├── overexpose/            # 單項：過曝（EV+1.2）
├── underexpose/           # 單項：欠曝（EV-1.2）
├── low_contrast/          # 單項：低對比度（alpha=0.6）
├── high_contrast/         # 單項：高對比度（alpha=1.6）
├── fov_zoomin/            # 單項：放大視野（zoom=1.15）
├── fov_zoomout/           # 單項：縮小視野（zoom=0.85）
├── fov_shift/             # 單項：視野平移（tx=0.07, ty=0.05）
├── fov_rotate/            # 單項：視野旋轉（3 度）
├── wb_warm/               # 單項：色溫偏暖
├── wb_cool/               # 單項：色溫偏冷
├── vignette/              # 單項：鏡頭暗角
├── noise/                 # 單項：感光雜訊
├── motion_blur/           # 單項：運動模糊
└── compound/              # 複合擾動（見下方）
```

### 6.2 `compound/` 子資料夾結構

```
compound/
├── WS08_001_compound_00.jpg   # 第 0 種複合擾動
├── WS08_001_compound_01.jpg   # 第 1 種複合擾動
├── ...
├── WS08_002_compound_00.jpg
├── WS08_002_compound_01.jpg
├── ...
└── perturbation_params.json   # ← 所有擾動參數記錄
```

### 6.3 `perturbation_params.json` 格式範例

```json
{
  "WS08_001": [
    {
      "variant": 0,
      "seed": 1042,
      "applied": {
        "exposure":  {"ev_shift": 0.872},
        "fov":       {"zoom": 1.093, "tx": 0.031, "ty": -0.042, "rotation": 1.7},
        "noise":     {"std": 18.3}
      }
    },
    {
      "variant": 1,
      "seed": 2042,
      "applied": {
        "roi_blur":      {"intensity": 0.451},
        "white_balance": {"r_shift": 1.142, "b_shift": 0.903},
        "contrast":      {"alpha": 0.712, "beta": -8.5}
      }
    }
  ]
}
```

> 保留此 JSON 的用途：
> 1. 重現特定擾動（發現模型在某組合上失敗時，可用相同 seed 重現）
> 2. 分析「哪種擾動組合最容易讓模型失敗」

---

## 7. 搭配泛化性評估的完整流程

### 7.1 整體流程

```
Step 1：準備原始測試集圖像（帶有 ground truth 標注）
         dataset/test/WS08/images/   ← 原始圖像
         dataset/test/WS08/labels/   ← ground truth（不動）

Step 2：執行模擬腳本，產生擾動後圖像
         python simulate_machine_adjustment.py \
           --input dataset/test/WS08/images/ \
           --output output_simulated/WS08/ \
           --mode all --n_compounds 10

Step 3：對每個擾動子資料夾執行模型推論
         python evaluate_model.py --model yolo \
           --images output_simulated/WS08/overexpose/ \
           --labels dataset/test/WS08/labels/

Step 4：彙整各擾動類型的 Recall / AP50 下降幅度

Step 5：比較 YOLO 和 GroundingDINO 在各擾動下的表現差異
```

### 7.2 結果分析矩陣（報告用）

| 擾動類型 | YOLO Recall | GDino Recall | YOLO AP50 | GDino AP50 | 勝出方 |
|---|---|---|---|---|---|
| 基準（無擾動）| 0.88 | 0.85 | 0.84 | 0.77 | YOLO |
| 輕度模糊 | 0.82 | 0.83 | 0.78 | 0.75 | GDino |
| 重度模糊 | 0.61 | 0.74 | 0.55 | 0.67 | GDino |
| 過曝 | 0.72 | 0.78 | 0.66 | 0.71 | GDino |
| 欠曝 | 0.68 | 0.76 | 0.61 | 0.68 | GDino |
| 低對比 | 0.75 | 0.80 | 0.69 | 0.73 | GDino |
| FOV 縮放 | 0.80 | 0.82 | 0.74 | 0.75 | GDino |
| FOV 平移 | 0.77 | 0.83 | 0.72 | 0.76 | GDino |
| 色溫偏移 | 0.85 | 0.83 | 0.80 | 0.75 | YOLO |
| 複合擾動（均值）| 0.65 | 0.76 | 0.58 | 0.68 | GDino |

**預期結論模式**：
- YOLO 在基準（訓練分布內）效果略優
- GroundingDINO 在各種 domain shift 下的 Recall 下降幅度更小
- **複合擾動** 差距最明顯，最接近真實機台調整後的情況

### 7.3 視覺化建議（報告用圖表）

**雷達圖**：以 8 種擾動類型為軸，比較兩個模型的 Recall 保持率

```
            基準
            1.0
     色溫  ●─●  模糊
          /     \
  暗角  ●         ●  曝光
        |  YOLO   |
  雜訊  ●  ─ ─   ●  對比
          \     /
     旋轉  ●─●  平移
            FOV

（實線=YOLO，虛線=GroundingDINO）
```

---

## 8. 參數調整指南

### 8.1 如何決定擾動強度？

建議的校準流程：

```
1. 先用 intensity=0.5 / ev_shift=1.0 等中等強度執行
2. 視覺比對模擬圖與真實調機後圖像
3. 調整強度直到「肉眼看起來和實際調機後差不多」
4. 確認強度後固定參數，確保所有實驗使用相同設定
```

### 8.2 `PERTURBATION_RANGES` 調整範例

**情境：模擬光源更換（主要是曝光和色溫，不涉及機械位移）**

```python
PERTURBATION_RANGES = {
    "roi_blur":      {"enabled": False},
    "motion_blur":   {"enabled": False},
    "exposure":      {"enabled": True,  "ev_shift": (-1.8, 1.8)},
    "contrast":      {"enabled": True,  "alpha": (0.7, 1.5), "beta": (-15, 15)},
    "fov":           {"enabled": False},
    "white_balance": {"enabled": True,  "r_shift": (0.8, 1.3), "b_shift": (0.8, 1.3)},
    "vignette":      {"enabled": True,  "strength": (0.1, 0.4)},
    "noise":         {"enabled": False},
}
```

**情境：模擬治具重新定位（主要是視野和輕微模糊）**

```python
PERTURBATION_RANGES = {
    "roi_blur":      {"enabled": True,  "intensity": (0.1, 0.4)},
    "motion_blur":   {"enabled": False},
    "exposure":      {"enabled": False},
    "contrast":      {"enabled": False},
    "fov":           {"enabled": True,  "zoom": (0.9, 1.1), "tx": (-0.1, 0.1), "ty": (-0.1, 0.1), "rotation": (-4.0, 4.0)},
    "white_balance": {"enabled": False},
    "vignette":      {"enabled": False},
    "noise":         {"enabled": False},
}
```

### 8.3 `n_perturbations` 建議值

| 場景 | 建議值 | 說明 |
|---|---|---|
| 輕度調整（微調參數）| 1 ~ 2 | 模擬小幅度調機 |
| 標準調整（正常換料）| 3 | 預設值，最接近典型情況 |
| 嚴重調整（大改配置）| 4 ~ 5 | 模擬重大工站改造 |

---

## 9. 常見問題

### Q1：模擬圖像的 ground truth 標注要怎麼處理？

**A**：不需要修改標注。模擬的所有效果都是圖像層面的像素變換，**不改變物件的位置和大小**，所以原始的 ground truth bounding box 直接沿用即可。

唯一例外是 `simulate_fov_change`：若有縮放和平移，嚴格來說應該對 bounding box 做對應的幾何變換。簡單處理方式是只在泛化性分析時使用，不計算精確 AP，只看模型「有沒有找到物件」的 Recall。

---

### Q2：`compound` 模式下的結果每次都不一樣？

**A**：不會，只要 `--seed` 固定，結果就是可重現的。腳本為每張圖的每個 variant 計算獨立 seed：

```python
seed = base_seed + i * 1000 + hash(stem) % 1000
```

相同的 `--seed 42`、相同圖片名稱、相同 `i`（variant 編號），輸出必然相同。

---

### Q3：可以在 Label Studio 或 CVAT 中直接使用模擬圖像嗎？

**A**：可以。模擬輸出都是標準 JPEG，可以直接匯入任何標注工具。建議將原始圖像和模擬圖像分開匯入不同的 Project 或 Task Group，方便管理。

---

### Q4：如何把這個腳本整合進評估 pipeline？

**A**：建議的整合方式：

```python
from simulate_machine_adjustment import apply_compound_perturbation, PERTURBATION_RANGES
import cv2

# 在評估迴圈中動態產生擾動，不需要預先存檔
image = cv2.imread("WS08_001.jpg")

for seed in range(10):  # 10 種擾動
    augmented, params = apply_compound_perturbation(
        image, PERTURBATION_RANGES,
        n_perturbations=3,
        seed=seed
    )
    # 直接送入模型推論，不需要寫檔
    predictions = model.predict(augmented)
    # ... 計算 metrics
```

---

*文件版本：v1.0*
*最後更新：2026-03*
*對應腳本：`simulate_machine_adjustment.py`*
