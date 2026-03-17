# 少樣本工業瑕疵偵測 AI 方案完整分析報告

> 研究日期：2026-03-17
> 主題：DINOv3、SSVP 及相關基礎模型在工業視覺品質檢測的應用分析

---

## 目錄

1. [背景與使用場景定義](#1-背景與使用場景定義)
2. [DINOv3 基礎模型介紹](#2-dinov3-基礎模型介紹)
3. [核心方法論：Patch-level 相似度搜尋](#3-核心方法論patch-level-相似度搜尋)
4. [主要模型方案分析](#4-主要模型方案分析)
   - [4.1 AnomalyDINO（WACV 2025）](#41-anomalydino-wacv-2025)
   - [4.2 AD-DINOv3（2025）](#42-ad-dinov3-2025)
   - [4.3 FoundAD（2025）](#43-foundad-2025)
   - [4.4 SSVP（2026）](#44-ssvp-2026)
   - [4.5 CLIP-DINOv2 Multimodal Fusion（2025）](#45-clip-dinov2-multimodal-fusion-2025)
5. [傳統基線方法比較](#5-傳統基線方法比較)
6. [效能指標總覽對照表](#6-效能指標總覽對照表)
7. [ROI-based 應用整合方案](#7-roi-based-應用整合方案)
8. [實作建議與選型指南](#8-實作建議與選型指南)
9. [完整技術整合架構](#9-完整技術整合架構)
10. [參考資料](#10-參考資料)

---

## 1. 背景與使用場景定義

### 目標任務

在工業生產品質檢測場景中，需要：

- **少樣本（Few-shot）** 瑕疵偵測：僅提供 1～5 張參考圖（含 ROI 標注或 bounding box）
- **ROI-based 視覺相似度搜尋**：在指定感興趣區域內比對外觀差異
- **Pixel-level 異常定位**：不只判斷是否有瑕疵，還需標出位置
- **快速部署**：不需大量標注資料或長時間訓練

### 核心挑戰

| 挑戰 | 說明 |
|------|------|
| 標注稀缺 | 瑕疵樣本少、取得困難 |
| 類別多樣 | 每種產品瑕疵形態各異 |
| 細粒度差異 | 瑕疵區域小、與正常品外觀接近 |
| 即時性需求 | 生產線需要快速推論 |

---

## 2. DINOv3 基礎模型介紹

### 什麼是 DINOv3？

DINOv3 是 Meta AI 最新一代的 **自監督視覺基礎模型（Self-supervised Vision Foundation Model）**，是 DINOv2 的後繼版本，採用 Vision Transformer (ViT) 架構，在大規模無標注圖像資料上訓練。

### 核心特性

| 特性 | 說明 |
|------|------|
| **自監督學習** | 無需任何人工標注即可訓練 |
| **Patch-level 特徵** | 每個 16×16 pixel patch 都對應高品質語義向量 |
| **Register Tokens** | 抑制 outlier patch 活化，特徵分佈更穩定 |
| **RoPE 位置編碼** | 解析度無關，ROI 大小不固定也能處理 |
| **強泛化性** | 未見過的工業產品類別也能提取高品質特徵 |
| **多尺度結構先驗** | 層疊 Transformer 層捕捉從低到高層次的視覺結構 |

### 為何特別適合瑕疵偵測？

DINOv3 的 patch embedding 天生捕捉**局部紋理、形狀一致性**，而工業瑕疵恰好表現為：

```
正常 patch 特徵 ←→ 距離近（在特徵空間中聚集）
瑕疵 patch 特徵 ←→ 距離遠（偏離正常分佈）
```

這個特性讓 DINOv3 可以在**完全不訓練**的情況下，通過近鄰搜尋實現異常偵測。

---

## 3. 核心方法論：Patch-level 相似度搜尋

### 基本流程

```
┌─────────────────────────────────────────────────────┐
│                  參考樣本建庫階段                      │
│                                                     │
│  參考圖（1~N張） → [ROI裁切] → DINOv3               │
│                                ↓                    │
│                         Patch Embeddings            │
│                         [M × D 矩陣]                │
│                                ↓                    │
│                        Memory Bank 建立              │
└─────────────────────────────────────────────────────┘
                          ↓ (一次性)

┌─────────────────────────────────────────────────────┐
│                    推論偵測階段                        │
│                                                     │
│  測試圖 → [ROI裁切] → DINOv3 → Patch Embeddings    │
│                                    ↓                │
│                               KNN 距離搜尋           │
│                               (vs Memory Bank)      │
│                                    ↓                │
│                            每個 patch 的異常分數     │
│                                    ↓                │
│                        Anomaly Heatmap（熱力圖）     │
│                        + Image-level 異常分數        │
└─────────────────────────────────────────────────────┘
```

### ROI 整合示意

```python
# 概念示意（非完整實作）
import torch

model = DINOv3("vit-large")  # 凍結權重，無需訓練

# === 建庫階段 ===
memory_bank = []
for ref_img, roi_bbox in reference_samples:
    roi_crop = crop(ref_img, roi_bbox)           # 裁切 ROI
    patches = model.get_patch_embeddings(roi_crop)  # [N_patches, 1024]
    memory_bank.append(patches)
memory_bank = torch.cat(memory_bank, dim=0)     # [Total_patches, 1024]

# === 推論階段 ===
test_roi = crop(test_img, test_roi_bbox)
test_patches = model.get_patch_embeddings(test_roi)  # [N, 1024]
distances = knn_search(test_patches, memory_bank)     # [N] 每個 patch 的距離
anomaly_map = distances.reshape(h_patches, w_patches) # 重組為熱力圖
anomaly_score = anomaly_map.max()                      # 或 top-k 平均
```

---

## 4. 主要模型方案分析

### 4.1 AnomalyDINO (WACV 2025)

**定位：最成熟、最易部署的 DINOv2/v3 少樣本方案**

| 屬性 | 詳情 |
|------|------|
| 論文 | AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 |
| 發表 | WACV 2025 (IEEE/CVF Winter Conference) |
| arXiv | [2405.14529](https://arxiv.org/abs/2405.14529) |
| GitHub | [dammsi/AnomalyDINO](https://github.com/dammsi/AnomalyDINO) |
| 訓練需求 | **完全 training-free** |

#### 技術亮點

1. **Patch-level Deep Nearest Neighbor**：遵循成熟的 patch KNN 範式
2. **DINOv2 自身分割能力**：利用 DINOv2 的 attention map 做前景遮罩，不需額外分割模型
3. **One-shot 特化預處理**：針對單張參考圖設計數據增強策略
4. **雙重輸出**：同時提供 image-level 分類和 pixel-level 分割

#### 效能（MVTec-AD）

| 設定 | Image AUROC |
|------|------------|
| 1-shot | **96.6%** |
| 2-shot | 97.2% |
| 4-shot | 97.8% |

> 相較前代 1-shot AUROC 93.1% 提升至 96.6%，提升幅度顯著。

#### 適用場景

- 有 1~5 張參考圖（正常品或瑕疵品）
- 希望快速驗證，不想訓練模型
- 需要 pixel-level 瑕疵定位
- 計算資源有限（CPU 友善，支援 FAISS CPU 模式）

---

### 4.2 AD-DINOv3 (2025)

**定位：首個將 DINOv3 引入 Zero-shot 異常偵測的方案**

| 屬性 | 詳情 |
|------|------|
| 論文 | AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration |
| arXiv | [2509.14084](https://arxiv.org/abs/2509.14084) |
| GitHub | [Kaisor-Yuan/AD-DINOv3](https://github.com/Kaisor-Yuan/AD-DINOv3) |
| Backbone | DINOv3 ViT-L/16 + CLIP text encoder |
| 訓練需求 | 需要 auxiliary 資料集訓練（少量） |

#### 技術亮點

1. **多模態對比學習**：DINOv3 視覺 + CLIP 文字，形成視覺-語言聯合空間
2. **Lightweight Adapters**：在兩種模態上加輕量 adapter，彌合領域偏差
3. **Multi-level Feature Adaptation**：提取 DINOv3 第 6、12、18、24 層特徵，融合多尺度資訊
4. **Anomaly-Aware Calibration Module (AACM)**：
   - 引導 CLS token 關注異常區域而非全域語義
   - 使用 focal loss（難樣本加強）+ Dice loss（空間一致性）

#### 效能對比（工業 + 醫療 8 個 benchmark）

| 方法 | 工業 AUROC | 工業 F1 | 醫療 AUROC | 醫療 F1 |
|------|-----------|---------|-----------|---------|
| WinCLIP | 77.1% | 20.1% | - | - |
| AnomalyCLIP | 94.2% | 37.5% | - | - |
| **AD-DINOv3** | **94.2%** | **44.6%** | **84.5%** | **51.5%** |

> 與 AnomalyCLIP 在 AUROC 相當，但 F1 分數顯著更高（44.6% vs 37.5%），代表更少誤報。

#### 適用場景

- Zero-shot：不提供任何該類別樣本，只給文字描述
- 工業 + 醫療雙領域都需要覆蓋
- 對 F1（precision-recall 平衡）有要求

---

### 4.3 FoundAD (2025)

**定位：最精簡的 DINOv3 Few-shot 異常偵測方案**

| 屬性 | 詳情 |
|------|------|
| 論文 | Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors |
| arXiv | [2510.01934](https://arxiv.org/abs/2510.01934) |
| GitHub | [ymxlzgy/FoundAD](https://github.com/ymxlzgy/FoundAD) |
| 機構 | Technical University of Munich + MVTec Software GmbH |
| Backbone | DINOv3（及多種基礎編碼器） |

#### 核心洞察

> **「異常量與學習到的 embedding 差異直接相關」**

正常圖像的特徵應該落在「自然圖像流形（natural image manifold）」上，異常圖像的特徵則偏離這個流形。FoundAD 學習一個**非線性投影算子**把特徵映射回流形，偏差即為異常分數。

#### 技術亮點

1. **非線性流形投影**：輕量 projector 網路，參數遠少於其他方法
2. **Multi-class 偵測**：單一模型同時處理多種產品類別
3. **背景雜訊抑制**：分割圖更乾淨，背景誤報少
4. **DINOv3 加成**：使用 DINOv3 作為 backbone 時達到最佳效能

#### Few-shot 效能（MVTec-AD）

| 設定 | 效能 |
|------|------|
| 1-shot | 競爭性 SOTA |
| 2-shot | 持續提升 |
| 4-shot | 超越所有 class-specific 方法 |

> 以更少參數量超越針對單一類別優化的方法。

#### 適用場景

- 需要同時偵測多種產品類別（multi-class）
- 對模型大小和推論效率有要求
- 可提供少量樣本做輕量訓練的場景

---

### 4.4 SSVP (2026)

**定位：當前 Zero-shot 工業異常偵測的最強方案**

| 屬性 | 詳情 |
|------|------|
| 論文 | SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot Anomaly Detection |
| arXiv | [2601.09147](https://arxiv.org/abs/2601.09147) |
| 發表日期 | 2026 年 1 月 14 日 |
| 作者 | Chenhao Fu 等 |
| Backbone | CLIP ViT-L/14 + DINOv3 ViT-L/16（雙編碼器） |
| 訓練環境 | NVIDIA RTX 4090 |

#### 架構設計

```
輸入圖片
   │
   ├──────────────────────────────────────────┐
   ↓                                          ↓
CLIP ViT-L/14                           DINOv3 ViT-L/16
（全域語義特徵）                          （細粒度結構特徵）
   │                                          │
   └──────────────┬───────────────────────────┘
                  ↓
    ┌─────────────────────────────┐
    │  HSVS                       │
    │  Hierarchical Semantic-     │
    │  Visual Synergy             │
    │                             │
    │  DINOv3 多尺度結構先驗      │
    │  → 注入 CLIP 語義空間       │
    │  (via 可學習投影矩陣)        │
    └─────────────────────────────┘
                  ↓
    ┌─────────────────────────────┐
    │  VCPG                       │
    │  Vision-Conditioned         │
    │  Prompt Generator           │
    │                             │
    │  Cross-modal attention      │
    │  → 動態生成文字 prompt      │
    │  → 精確對應特定異常模式     │
    └─────────────────────────────┘
                  ↓
    ┌─────────────────────────────┐
    │  VTAM                       │
    │  Visual-Text                │
    │  Anomaly Mapper             │
    │                             │
    │  雙閘門校準範式             │
    │  → 對齊全域評分與局部證據   │
    └─────────────────────────────┘
                  ↓
         Anomaly Score + Map
```

#### 三大核心模組

**1. HSVS（Hierarchical Semantic-Visual Synergy）**
- Adaptive Token Features Fusion (ATF) block
- 雙路徑跨模態注意力對齊不同特徵
- 顯式注入 DINOv3 的多尺度結構先驗

**2. VCPG（Vision-Conditioned Prompt Generator）**
- 跨模態注意力引導動態 prompt 生成
- 語言查詢精確錨定特定異常模式
- 解決固定文字 prompt 泛化性不足的問題

**3. VTAM（Visual-Text Anomaly Mapper）**
- 雙閘門校準：解決全域評分與局部證據不一致問題
- Margin-based 正則化：防止語義漂移

#### 效能（7 個工業 benchmark）

| 指標 | 分數 |
|------|------|
| MVTec-AD Image-AUROC | **93.0%** |
| MVTec-AD Pixel-AUROC | **92.2%** |
| TIR-FG（相對基線提升） | **+28.88%** |
| 訓練輪次 | 15 epochs |

**Backbone 消融實驗**：
- DINOv3（自監督）: 93.0% Image-AUROC
- MAE（重建式）: 90.5% Image-AUROC
- → DINOv3 結構先驗優於重建式預訓練

**規模消融**：DINOv3 從 ViT-Small 擴展至 ViT-Large 有持續提升。

#### 局限性

- **雙 backbone 計算開銷大**：推論速度受限，不適合邊緣即時部署
- 需要 auxiliary 訓練（15 epochs）

#### 適用場景

- Zero-shot：未見類別無樣本也可偵測
- 高精度要求的離線品質分析
- 具備 GPU 計算資源的場景

---

### 4.5 CLIP-DINOv2 Multimodal Fusion (2025)

| 屬性 | 詳情 |
|------|------|
| 論文 | Zero-Shot Industrial Anomaly Detection via CLIP-DINOv2 Multimodal Fusion and Stabilized Attention Pooling |
| 發表 | MDPI Electronics, 2025年12月 |
| 連結 | [mdpi.com/2079-9292/14/24/4785](https://www.mdpi.com/2079-9292/14/24/4785) |

#### 核心技術

**Dual-Modality Attention (DMA) Mechanism**：
- CLIP 全域語義表示（宏觀結構異常）
- DINOv2 多尺度局部結構特徵（微觀紋理偏差）
- 同時捕捉兩種尺度的異常

#### 效能（7 個工業 benchmark 平均）

| 指標 | 分數 |
|------|------|
| Image-level AUROC | **93.4%** |
| Pixel-level AUROC | **96.9%** |

> 超越 WinCLIP、AnomalyCLIP、AdaCLIP 等全部先前 zero-shot 方法。

---

## 5. 傳統基線方法比較

| 方法 | 年份 | 典範 | MVTec-AD AUROC | 需要訓練資料 |
|------|------|------|---------------|------------|
| **PatchCore** | 2022 | Full-shot memory bank | 99.6% | 是（大量正常品） |
| **WinCLIP** | CVPR 2023 | Zero-shot CLIP | 91.8% | 否 |
| **WinCLIP+** | CVPR 2023 | 1-shot CLIP | 93.1% | 否（1張） |
| **AnomalyCLIP** | ICLR 2024 | Zero-shot + aux | 94.2% | 是（aux 資料集） |

### PatchCore 的問題

PatchCore 雖然精度最高（99.6%），但需要**每種產品類別提供大量正常品**建立完整記憶庫，不適合：
- 新產品快速驗證
- 樣本稀缺的場景
- 多樣化產品線

---

## 6. 效能指標總覽對照表

### MVTec-AD 基準（Image-level AUROC）

| 方法 | 範式 | Image AUROC | Pixel AUROC | 樣本需求 | 訓練 | 速度 |
|------|------|------------|------------|---------|------|------|
| PatchCore | Full-shot | 99.6% | 98.2% | 大量 | 否* | 快 |
| WinCLIP | Zero-shot | 91.8% | 85.1% | 0 | 否 | 快 |
| WinCLIP+ | 1-shot | 93.1% | 95.2% | 1 | 否 | 快 |
| AnomalyCLIP | Zero-shot | 94.2% | ~93% | 0+aux | 是 | 快 |
| **AnomalyDINO** | **1-shot** | **96.6%** | **~95%** | **1** | **否** | **快** |
| AD-DINOv3 | Zero-shot | 94.2% | - | 0+aux | 是 | 中 |
| CLIP-DINOv2 | Zero-shot | 93.4% | 96.9% | 0 | 是 | 中 |
| **SSVP** | **Zero-shot** | **93.0%** | **92.2%** | **0** | **是** | **慢** |
| FoundAD (DINOv3) | Few-shot | SOTA | SOTA | 1~4 | 輕量 | 快 |

> *PatchCore 不需梯度訓練，但需要完整 memory bank 建立

---

## 7. ROI-based 應用整合方案

### 場景假設

```
輸入：一張待測圖 + ROI bounding box (x, y, w, h)
參考：1~5 張已知正常/瑕疵圖 + 對應 ROI 標注
輸出：
  - anomaly_score: 0~1 浮點數
  - anomaly_map: pixel-level 熱力圖
  - decision: PASS / FAIL
```

### 方案 A：AnomalyDINO + ROI 裁切（推薦入門方案）

```
優點：training-free、1-shot 可用、有 GitHub 程式碼
缺點：需要提供參考圖

實作步驟：
1. 安裝 AnomalyDINO（github.com/dammsi/AnomalyDINO）
2. 載入 DINOv2/DINOv3 backbone
3. 對所有參考圖做 ROI 裁切，建立 memory bank
4. 對測試圖做 ROI 裁切後推論，取得 anomaly map
5. 設定閾值進行 PASS/FAIL 判定
```

### 方案 B：FoundAD + 自訂 ROI pipeline（推薦進階方案）

```
優點：DINOv3 backbone、multi-class 支援、輕量訓練
缺點：需要少量訓練（但不需要瑕疵標注）

實作步驟：
1. 安裝 FoundAD（github.com/ymxlzgy/FoundAD）
2. 準備各類別 ROI 裁切的正常圖（10~50張）
3. 訓練輕量 projector（幾分鐘）
4. 推論時裁切 ROI 後輸入模型
5. 偏離流形的距離即為異常分數
```

### 方案 C：SSVP（最高精度方案）

```
優點：Zero-shot SOTA、不需參考圖
缺點：雙 backbone 計算量大、需 GPU

適用：離線品質分析、需要文字描述驅動的場景
```

### 方案 D：自建 DINOv3 Patch Search（最大彈性）

```python
# 最直接的 DINOv3 patch 相似度搜尋
import torch
import faiss
import numpy as np

class DINOv3DefectDetector:
    def __init__(self, backbone="vit_large"):
        self.model = load_dinov3(backbone)  # 凍結
        self.index = None

    def build_reference(self, ref_images, roi_bboxes):
        """建立 reference memory bank"""
        all_patches = []
        for img, bbox in zip(ref_images, roi_bboxes):
            roi = self._crop_roi(img, bbox)
            patches = self._extract_patches(roi)  # [N, D]
            all_patches.append(patches)

        all_patches = np.vstack(all_patches)  # [Total, D]

        # FAISS 索引加速 KNN 搜尋
        self.index = faiss.IndexFlatL2(all_patches.shape[1])
        self.index.add(all_patches.astype(np.float32))

    def predict(self, test_img, roi_bbox):
        """推論：輸出異常分數和熱力圖"""
        roi = self._crop_roi(test_img, roi_bbox)
        test_patches = self._extract_patches(roi)  # [N, D]

        # KNN 距離搜尋
        k = 1
        distances, _ = self.index.search(test_patches.astype(np.float32), k)
        anomaly_scores = distances[:, 0]  # [N]

        # 重組為熱力圖
        h_patches = roi.shape[0] // 16
        w_patches = roi.shape[1] // 16
        anomaly_map = anomaly_scores.reshape(h_patches, w_patches)

        # 上採樣到原始解析度
        anomaly_map = upsample(anomaly_map, roi.shape[:2])

        return {
            "anomaly_score": float(anomaly_scores.max()),
            "anomaly_map": anomaly_map,
            "decision": "FAIL" if anomaly_scores.max() > self.threshold else "PASS"
        }

    def _extract_patches(self, image):
        with torch.no_grad():
            tokens = self.model.get_intermediate_layers(image, n=1)[0]
            # 去掉 CLS token，保留 patch tokens
            patch_tokens = tokens[:, 1:, :]
        return patch_tokens.squeeze(0).cpu().numpy()
```

---

## 8. 實作建議與選型指南

### 決策樹

```
你有幾張參考圖？
│
├─ 0張（只有文字描述）
│   └─ → SSVP 或 AD-DINOv3（Zero-shot）
│
├─ 1~5張
│   ├─ 不想訓練 → AnomalyDINO（Training-free）
│   └─ 可接受輕量訓練 → FoundAD（DINOv3 backbone）
│
└─ 10張以上
    ├─ 多個產品類別 → FoundAD（Multi-class）
    └─ 單一類別、高精度 → 自建 DINOv3 + FAISS
```

### 各方案適用場景總結

| 使用場景 | 推薦方案 | 理由 |
|---------|---------|------|
| 新產品快速驗證（1~3張圖） | **AnomalyDINO** | 零訓練、立即可用 |
| 多產品線統一管理 | **FoundAD** | Multi-class、輕量 |
| 完全無樣本、文字驅動 | **SSVP** | Zero-shot SOTA |
| ROI 精確比對 | **DINOv3 Patch Search** | 最大彈性控制 |
| 邊緣 / 即時部署 | **AnomalyDINO** | 單 backbone、FAISS CPU |
| 高精度離線分析 | **SSVP 或 AD-DINOv3** | 雙模態融合 |
| Pixel-level 精確定位 | **任意 DINOv3 方案** | 天生 patch-level |

### 硬體需求參考

| 方案 | 最低 GPU | 推薦 GPU | CPU 可用 |
|------|---------|---------|---------|
| AnomalyDINO | GTX 1080 | RTX 3090 | 是（較慢） |
| FoundAD | RTX 3090 | RTX 4090 | 部分支援 |
| AD-DINOv3 | RTX 3090 | RTX 4090 | 否 |
| SSVP | RTX 4090 | A100 | 否 |
| 自建 DINOv3 | GTX 1080 | RTX 3090 | 是（FAISS） |

---

## 9. 完整技術整合架構

### 生產環境建議架構

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 / 介面層                           │
│                                                             │
│   相機/影像輸入 → ROI 框選工具 → 參考圖管理介面              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                     前處理層                                 │
│                                                             │
│   ROI 裁切 → 解析度正規化 → 色彩空間轉換                    │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   DINOv3 特徵提取層                          │
│                                                             │
│   DINOv3 ViT-L/16（凍結）                                   │
│   → Patch Embeddings [N × 1024]                            │
│   → 多層特徵融合（層 6, 12, 18, 24）                        │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    異常偵測層                                 │
│                                                             │
│   Option A: FAISS KNN（最近鄰距離）                         │
│   Option B: FoundAD projector（流形距離）                   │
│   Option C: SSVP（CLIP 文字引導）                           │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    後處理 + 決策層                            │
│                                                             │
│   Anomaly Map 上採樣 → 高斯平滑                             │
│   → 閾值判定 → PASS/FAIL                                    │
│   → 視覺化標注輸出                                           │
└─────────────────────────────────────────────────────────────┘
```

### 建議開發流程

```
第 1 週：
  - 安裝 AnomalyDINO，在 MVTec-AD 驗證流程
  - 整合 ROI 裁切 pipeline
  - 收集第一批產品參考圖（5~10 張）

第 2 週：
  - 對真實產品數據測試 AnomalyDINO
  - 調整閾值，評估 precision / recall
  - 比較 1-shot vs 3-shot vs 5-shot 效能差異

第 3~4 週：
  - 若需要更高精度：評估 FoundAD（DINOv3 backbone）
  - 若需要多類別：測試 FoundAD multi-class 模式
  - 效能滿足需求則停止，不需再引入更複雜模型

長期：
  - 根據生產數據建立更完整的 memory bank
  - 定期更新 reference samples
  - 考慮 SSVP 作為難樣本補充判定
```

---

## 10. 參考資料

### 核心論文

| 論文 | 年份 | 連結 |
|------|------|------|
| AnomalyDINO: Boosting Patch-based Few-shot Anomaly Detection with DINOv2 | WACV 2025 | [arXiv:2405.14529](https://arxiv.org/abs/2405.14529) |
| AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration | 2025 | [arXiv:2509.14084](https://arxiv.org/abs/2509.14084) |
| Foundation Visual Encoders Are Secretly Few-Shot Anomaly Detectors (FoundAD) | 2025 | [arXiv:2510.01934](https://arxiv.org/abs/2510.01934) |
| SSVP: Synergistic Semantic-Visual Prompting for Industrial Zero-Shot Anomaly Detection | 2026 | [arXiv:2601.09147](https://arxiv.org/abs/2601.09147) |
| Zero-Shot Industrial Anomaly Detection via CLIP-DINOv2 Multimodal Fusion | 2025 | [MDPI Electronics](https://www.mdpi.com/2079-9292/14/24/4785) |
| WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | CVPR 2023 | [arXiv:2303.14539](https://arxiv.org/abs/2303.14539) |

### 開源程式碼

| 專案 | 連結 |
|------|------|
| AnomalyDINO (Official) | [github.com/dammsi/AnomalyDINO](https://github.com/dammsi/AnomalyDINO) |
| AD-DINOv3 (Official) | [github.com/Kaisor-Yuan/AD-DINOv3](https://github.com/Kaisor-Yuan/AD-DINOv3) |
| FoundAD (Official) | [github.com/ymxlzgy/FoundAD](https://github.com/ymxlzgy/FoundAD) |
| Awesome Industrial Anomaly Detection | [github.com/M-3LAB/awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) |
| WinCLIP (Reproduction) | [github.com/mala-lab/WinCLIP](https://github.com/mala-lab/WinCLIP) |

### 資料集

| 資料集 | 說明 | 用途 |
|--------|------|------|
| **MVTec-AD** | 15 類工業品，5354 張圖像，含 pixel-level 標注 | 主要工業異常 benchmark |
| **VisA** | 12 類，10821 張，更多細粒度異常 | 細粒度工業 benchmark |
| **BTAD** | 實際工廠場景 | 真實工業場景測試 |
| **MPDD** | 金屬零件缺陷 | 金屬製造場景 |

---

*報告版本：v1.0 | 研究日期：2026-03-17*
