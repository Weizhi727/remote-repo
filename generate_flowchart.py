"""
generate_flowchart.py
====================
生成「模型/算法優化閉環流程圖」並輸出為 PNG。

依賴: matplotlib
安裝: pip install matplotlib
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse
import sys

# ── 中文字型設定（依系統環境自動尋找可用字型）──────────────────────────
matplotlib.rcParams['font.sans-serif'] = [
    'WenQuanYi Zen Hei',
    'Noto Sans CJK TC', 'Noto Sans CJK SC',
    'PingFang TC', 'Microsoft YaHei', 'SimHei',
    'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans',
]
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_FILE = 'model_optimization_flowchart.png'

# ── 色盤 ────────────────────────────────────────────────────────────────
C = {
    'start':   '#1D3557',  # 深藍灰 — 開始/結束
    'proc':    '#457B9D',  # 鋼藍  — 主流程
    'dec':     '#E76F51',  # 橙紅  — 判斷/決策
    'auto':    '#2D6A4F',  # 深綠  — 自動化標註
    'confirm': '#6A0572',  # 深紫  — 開發人員確認
    'normal':  '#43AA8B',  # 翡翠綠 — 正常推論
    'result':  '#023E8A',  # 深海藍 — 最終結果
    'arrow':   '#4A4E69',  # 深紫灰 — 一般箭頭
    'no':      '#E63946',  # 紅    — 否/差異過大
    'yes':     '#2A9D8F',  # 青    — 是/正常
    'loop':    '#8338EC',  # 紫    — 閉環回路
}

# ── 畫布 ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#EDF4FB')
ax.set_facecolor('#EDF4FB')


# ══════════════════════════════════════════════════════════════════════
# 工具函式
# ══════════════════════════════════════════════════════════════════════

def rbox(cx, cy, w, h, text, color, tc='white', fs=9.5, zorder=3):
    """圓角矩形節點"""
    shadow = FancyBboxPatch(
        (cx - w/2 + 0.08, cy - h/2 - 0.08), w, h,
        boxstyle="round,pad=0.18",
        facecolor='#00000025', edgecolor='none', zorder=zorder - 1)
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.18",
        facecolor=color, edgecolor='white', linewidth=2.2, zorder=zorder)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold',
            multialignment='center', zorder=zorder + 1)


def oval(cx, cy, w, h, text, color, tc='white', fs=10):
    """橢圓節點（開始/結束）"""
    e = Ellipse((cx, cy), w, h,
                facecolor=color, edgecolor='white', linewidth=2.5, zorder=3)
    ax.add_patch(e)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, color=tc, fontweight='bold', zorder=4)


def diamond(cx, cy, w, h, text, color, tc='white', fs=8.5):
    """菱形節點（判斷）"""
    pts = [[cx, cy + h/2], [cx + w/2, cy],
           [cx, cy - h/2], [cx - w/2, cy]]
    d = plt.Polygon(pts, facecolor=color, edgecolor='white',
                    linewidth=2.5, zorder=3)
    ax.add_patch(d)
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold', multialignment='center', zorder=4)


def arrow(x1, y1, x2, y2, color=None):
    """單段直線箭頭"""
    c = color or C['arrow']
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=c, lw=2.0,
                        mutation_scale=15), zorder=5)


def bent_arrow(points, color=None):
    """折線箭頭：前段為直線，最後一段帶箭頭"""
    c = color or C['arrow']
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i + 1][0]],
                [points[i][1], points[i + 1][1]],
                color=c, lw=2.0, zorder=5, solid_capstyle='round')
    ax.annotate(
        '', xy=points[-1], xytext=points[-2],
        arrowprops=dict(arrowstyle='->', color=c, lw=2.0,
                        mutation_scale=15), zorder=5)


def lbl(x, y, text, color=None, fs=8.5, ha='center'):
    """帶白底的文字標籤"""
    c = color or C['arrow']
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs,
            color=c, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.88, boxstyle='round,pad=0.18'),
            zorder=7)


# ══════════════════════════════════════════════════════════════════════
# 標題
# ══════════════════════════════════════════════════════════════════════
fig.text(
    0.5, 0.977,
    '模型 / 算法優化  閉環流程圖',
    ha='center', va='top', fontsize=17, fontweight='bold',
    color='white',
    bbox=dict(facecolor=C['start'], edgecolor='none',
              boxstyle='round,pad=0.4', alpha=0.95))
fig.text(
    0.5, 0.963,
    'Model / Algorithm Optimization  Closed-Loop Process',
    ha='center', va='top', fontsize=10, color=C['proc'], style='italic')


# ══════════════════════════════════════════════════════════════════════
# 節點繪製
# ══════════════════════════════════════════════════════════════════════

# ── 主流程（中央 x=10）────────────────────────────────────────────────
oval(10, 26.3, 3.6, 0.85, '開  始', C['start'])
rbox(10, 25.0, 4.6, 0.9,  '算法開發', C['proc'])
rbox(10, 23.6, 4.6, 0.9,  '模型上線部署', C['proc'])
rbox(10, 22.2, 4.6, 0.9,  '設備上拋圖像  進行檢測', C['proc'])

# 決策一：訓練清單
#   頂點位置：top(10,21.1) bottom(10,19.5) left(7.1,20.3) right(12.9,20.3)
diamond(10, 20.3, 5.8, 1.6,
        '設備是否在\n訓練清單內？', C['dec'])

# ── 左側分支（否 / 不在清單，x=3.5）────────────────────────────────
rbox(3.5, 18.5, 3.9, 0.95, '自動化標註\n與訓練流程', C['auto'], fs=9)
rbox(3.5, 16.7, 3.9, 0.95, '開發人員確認\n模型品質', C['confirm'], fs=9)
rbox(3.5, 14.9, 3.9, 0.95, '更新模型\n推論上線', C['proc'], fs=9)

# ── 右側分支（是 / 在清單，x=14）────────────────────────────────────
# 決策二：圖像質量差異
#   頂點：top(14,19.45) bottom(14,17.75) left(11.75,18.6) right(16.25,18.6)
diamond(14, 18.6, 4.5, 1.7,
        '圖像質量指標\n差異是否過大？', C['dec'], fs=8.2)

# 差異過大 → 向下
rbox(14, 16.5, 4.1, 0.95, '自動化標註與訓練\n增加模型泛化性', C['auto'], fs=8.5)
rbox(14, 14.7, 4.1, 0.95, '開發人員確認\n模型品質', C['confirm'], fs=9)
rbox(14, 12.9, 4.1, 0.95, '更新模型\n推論上線', C['proc'], fs=9)

# 差異不大 → 向右
rbox(18.5, 18.6, 2.8, 0.95, '正常推論\n檢測輸出', C['normal'], fs=9)

# ── 收斂節點 ────────────────────────────────────────────────────────
rbox(10, 11.0, 5.8, 1.05,
     '減少過殺  ▸  提升整體檢測準確率', C['result'], fs=9.5)
oval(10, 9.6, 4.8, 0.9, '閉環優化  持續改進', C['start'])


# ══════════════════════════════════════════════════════════════════════
# 箭頭繪製
# ══════════════════════════════════════════════════════════════════════

# 主流程
arrow(10, 25.88, 10, 25.46)           # 開始 → 算法開發
arrow(10, 24.56, 10, 24.07)           # 算法開發 → 上線部署
arrow(10, 23.17, 10, 22.67)           # 上線部署 → 設備上拋
arrow(10, 21.73, 10, 21.13)           # 設備上拋 → 決策一

# 決策一 → 否（左側）
bent_arrow([(7.1, 20.3), (3.5, 20.3), (3.5, 18.98)], color=C['no'])
lbl(5.1, 20.55, '  否  ', color=C['no'])

# 決策一 → 是（右側）
bent_arrow([(12.9, 20.3), (14.0, 20.3), (14.0, 19.45)], color=C['yes'])
lbl(13.6, 20.55, '  是  ', color=C['yes'])

# 左側分支
arrow(3.5, 18.03, 3.5, 17.18)         # 自動標註 → 開發確認
arrow(3.5, 16.23, 3.5, 15.38)         # 開發確認 → 更新推論

# 決策二 → 差異過大（向下）
arrow(14, 17.75, 14, 16.98)
lbl(15.05, 17.35, '差異過大', color=C['no'], fs=8)

# 決策二 → 差異不大（向右）
arrow(16.25, 18.6, 17.1, 18.6)
lbl(16.65, 18.92, '差異不大', color=C['yes'], fs=8)

# 差異過大分支
arrow(14, 16.03, 14, 15.18)           # 自動標註 → 開發確認
arrow(14, 14.23, 14, 13.38)           # 開發確認 → 更新推論

# ── 各分支匯入收斂節點（y=11.0） ──────────────────────────────────
# B1 邊界：x[7.1,12.9]  y[10.48,11.53]

# L3（更新推論，左）→ 收斂左側
bent_arrow(
    [(3.5, 14.43), (3.5, 11.0), (7.1, 11.0)],
    color=C['arrow'])

# R2c（更新推論，差異過大）→ 收斂右側（上緣）
bent_arrow(
    [(14, 12.43), (14, 11.48), (12.9, 11.48)],
    color=C['arrow'])

# R3（正常推論）→ 收斂右側（下緣）
bent_arrow(
    [(18.5, 18.13), (18.5, 10.58), (12.9, 10.58)],
    color=C['arrow'])

# 收斂 → 閉環
arrow(10, 10.48, 10, 10.07)

# ── 閉環回路（B2 底部 → 繞回設備上拋左側）──────────────────────
# 設備上拋 left edge x = 10 - 2.3 = 7.7
bent_arrow(
    [(10, 9.15), (10, 8.4),
     (1.2, 8.4), (1.2, 22.2), (7.7, 22.2)],
    color=C['loop'])
lbl(1.8, 15.3, '持\n續\n優\n化\n閉\n環', color=C['loop'], fs=9)


# ══════════════════════════════════════════════════════════════════════
# 步驟說明標籤（右側補充說明）
# ══════════════════════════════════════════════════════════════════════
notes = [
    (20.0, 20.3,  '① 新設備不在清單 → 觸發\n   自動化標註+訓練管線'),
    (20.0, 18.6,  '② 舊設備曝光/參數異常 →\n   圖像質量指標評估'),
    (20.0, 11.0,  '③ 兩路徑皆回歸閉環\n   系統性減少過殺'),
]
for nx, ny, nt in notes:
    ax.text(nx - 0.3, ny, nt, ha='right', va='center', fontsize=7.5,
            color='#333355', style='italic',
            bbox=dict(facecolor='#FFFFFFCC', edgecolor='#AAAACC',
                      boxstyle='round,pad=0.25', linewidth=0.8))


# ══════════════════════════════════════════════════════════════════════
# 圖例
# ══════════════════════════════════════════════════════════════════════
legend_items = [
    (C['proc'],    '主流程節點'),
    (C['dec'],     '判斷 / 決策'),
    (C['auto'],    '自動化標註與訓練'),
    (C['confirm'], '開發人員確認'),
    (C['normal'],  '正常推論輸出'),
    (C['result'],  '最終收斂結果'),
    (C['loop'],    '閉環回路'),
]
lx, ly0 = 0.4, 8.0
ax.text(lx + 0.65, ly0 + 0.6, '圖  例', ha='center', fontsize=9,
        fontweight='bold', color='#1A1A2E')
for i, (clr, lbl_text) in enumerate(legend_items):
    ly = ly0 - i * 0.72
    rbox(lx + 0.35, ly, 0.6, 0.42, '', clr, fs=1, zorder=3)
    ax.text(lx + 0.75, ly, lbl_text, va='center', fontsize=8, color='#1A1A2E')


# ══════════════════════════════════════════════════════════════════════
# 存檔
# ══════════════════════════════════════════════════════════════════════
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight', facecolor='#EDF4FB')
print(f"✅ 流程圖已儲存：{OUTPUT_FILE}")
plt.close()
