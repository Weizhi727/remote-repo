"""
generate_ppt.py
===============
生成「模型/算法優化閉環流程」PowerPoint 簡報。

依賴:
  pip install python-pptx matplotlib

使用方式:
  # 先產生流程圖圖片
  python generate_flowchart.py

  # 再產生 PPT
  python generate_ppt.py
"""

import os
import subprocess
import sys
from pathlib import Path

# ── 自動安裝缺少的套件 ──────────────────────────────────────────────
def ensure_packages(*pkgs):
    import importlib
    for pkg, import_name in pkgs:
        try:
            importlib.import_module(import_name)
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

ensure_packages(('python-pptx', 'pptx'), ('matplotlib', 'matplotlib'))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm
import io

# ── 常數 ───────────────────────────────────────────────────────────
FLOWCHART_IMG = 'model_optimization_flowchart.png'
OUTPUT_PPT    = 'model_optimization_flowchart.pptx'

# 16:9 投影片尺寸
SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

# 色盤（與流程圖統一）
RGB = {
    'dark_blue':  RGBColor(0x1D, 0x35, 0x57),
    'steel_blue': RGBColor(0x45, 0x7B, 0x9D),
    'light_blue': RGBColor(0xA8, 0xDA, 0xDC),
    'orange':     RGBColor(0xE7, 0x6F, 0x51),
    'green':      RGBColor(0x2D, 0x6A, 0x4F),
    'light_green':RGBColor(0x52, 0xB7, 0x88),
    'purple':     RGBColor(0x6A, 0x05, 0x72),
    'teal':       RGBColor(0x43, 0xAA, 0x8B),
    'red':        RGBColor(0xE6, 0x39, 0x46),
    'white':      RGBColor(0xFF, 0xFF, 0xFF),
    'bg':         RGBColor(0xED, 0xF4, 0xFB),
    'text':       RGBColor(0x1A, 0x1A, 0x2E),
    'gray':       RGBColor(0x6C, 0x75, 0x7D),
}


# ══════════════════════════════════════════════════════════════════════
# 輔助函式
# ══════════════════════════════════════════════════════════════════════

def set_slide_bg(slide, rgb: RGBColor):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = rgb


def add_textbox(slide, left, top, width, height,
                text, font_size=18, bold=False, color=None,
                align=PP_ALIGN.LEFT, italic=False):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    return txBox


def add_rect(slide, left, top, width, height,
             fill_rgb, border_rgb=None, border_width=None):
    from pptx.util import Pt as UPt
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_rgb
    if border_rgb:
        shape.line.color.rgb = border_rgb
        if border_width:
            shape.line.width = border_width
    else:
        shape.line.fill.background()
    return shape


def add_rect_with_text(slide, left, top, width, height,
                       text, fill_rgb, text_rgb=None,
                       font_size=14, bold=True, align=PP_ALIGN.CENTER):
    shape = add_rect(slide, left, top, width, height, fill_rgb)
    shape.text_frame.word_wrap = True
    tf = shape.text_frame
    tf.paragraphs[0].alignment = align
    run = tf.paragraphs[0].add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    if text_rgb:
        run.font.color.rgb = text_rgb
    else:
        run.font.color.rgb = RGB['white']
    # Vertical center
    from pptx.enum.text import MSO_ANCHOR
    tf.auto_size = None
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return shape


def add_bullet_slide_content(slide, items, left, top, width, height,
                             font_size=16, color=None, spacing_before=8):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, (bullet, text) in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(spacing_before)
        run = p.add_run()
        run.text = f'{bullet}  {text}'
        run.font.size = Pt(font_size)
        if color:
            run.font.color.rgb = color


# ══════════════════════════════════════════════════════════════════════
# 確保流程圖圖片存在
# ══════════════════════════════════════════════════════════════════════

if not Path(FLOWCHART_IMG).exists():
    print(f"⚠ 找不到 {FLOWCHART_IMG}，正在執行 generate_flowchart.py ...")
    result = subprocess.run([sys.executable, 'generate_flowchart.py'])
    if result.returncode != 0:
        print("❌ generate_flowchart.py 執行失敗，請手動執行後再試。")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════
# 建立簡報
# ══════════════════════════════════════════════════════════════════════

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]   # 完全空白版面


# ══════════════════════════════════════════════════════════════════════
# Slide 1 — 封面
# ══════════════════════════════════════════════════════════════════════

slide1 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide1, RGB['bg'])

# 深藍色裝飾帶（上方）
add_rect(slide1,
         left=Inches(0), top=Inches(0),
         width=SLIDE_W, height=Inches(0.18),
         fill_rgb=RGB['dark_blue'])

# 深藍色裝飾帶（下方）
add_rect(slide1,
         left=Inches(0), top=Inches(7.32),
         width=SLIDE_W, height=Inches(0.18),
         fill_rgb=RGB['dark_blue'])

# 中央底色塊
add_rect(slide1,
         left=Inches(1.5), top=Inches(1.8),
         width=Inches(10.33), height=Inches(3.5),
         fill_rgb=RGB['dark_blue'])

# 主標題
add_textbox(slide1,
            left=Inches(1.5), top=Inches(2.0),
            width=Inches(10.33), height=Inches(1.2),
            text='模型 / 算法優化  閉環流程',
            font_size=36, bold=True,
            color=RGB['white'], align=PP_ALIGN.CENTER)

# 副標題
add_textbox(slide1,
            left=Inches(1.5), top=Inches(3.2),
            width=Inches(10.33), height=Inches(0.7),
            text='Model / Algorithm Optimization  Closed-Loop Process',
            font_size=18, bold=False, italic=True,
            color=RGB['light_blue'], align=PP_ALIGN.CENTER)

# 說明文字
add_textbox(slide1,
            left=Inches(1.5), top=Inches(3.9),
            width=Inches(10.33), height=Inches(0.6),
            text='自動化標註 ▸ 閉環訓練 ▸ 持續減少過殺',
            font_size=15, bold=False,
            color=RGB['light_green'], align=PP_ALIGN.CENTER)

# 日期 / 版本
add_textbox(slide1,
            left=Inches(0.3), top=Inches(6.9),
            width=Inches(5), height=Inches(0.4),
            text='Algorithm Development Team',
            font_size=11, color=RGB['gray'])


# ══════════════════════════════════════════════════════════════════════
# Slide 2 — 問題背景
# ══════════════════════════════════════════════════════════════════════

slide2 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide2, RGB['bg'])

# 頁首條
add_rect(slide2,
         left=Inches(0), top=Inches(0),
         width=SLIDE_W, height=Inches(1.1),
         fill_rgb=RGB['dark_blue'])
add_textbox(slide2,
            left=Inches(0.4), top=Inches(0.15),
            width=Inches(12), height=Inches(0.8),
            text='問題背景 — 為何需要閉環優化？',
            font_size=24, bold=True,
            color=RGB['white'])

# 問題卡片
problems = [
    ('🔺', '新設備圖像不在訓練集',
     '新型號設備上線時，其圖像風格與過往訓練資料差異大，導致模型誤判（過殺率上升）。',
     RGB['orange']),
    ('📷', '曝光/拍攝參數變動',
     '既有設備曝光調整後，圖像亮度、對比等質量指標發生偏移，超出模型適應範圍。',
     RGB['steel_blue']),
    ('⏳', '依賴人工反饋，週期過長',
     '過往僅靠用戶反饋才能觸發優化，從問題發現到修復往往需要數週甚至更長時間。',
     RGB['purple']),
]

card_w = Inches(3.8)
card_h = Inches(4.8)
gap    = Inches(0.4)
start_x = Inches(0.55)
start_y = Inches(1.4)

for i, (icon, title, body, clr) in enumerate(problems):
    cx = start_x + i * (card_w + gap)

    # 卡片背景
    add_rect(slide2, cx, start_y, card_w, card_h, clr)

    # 圖示
    add_textbox(slide2,
                left=cx + Inches(0.1), top=start_y + Inches(0.15),
                width=card_w - Inches(0.2), height=Inches(0.7),
                text=icon, font_size=30,
                color=RGB['white'], align=PP_ALIGN.CENTER)

    # 小標
    add_textbox(slide2,
                left=cx + Inches(0.15), top=start_y + Inches(0.85),
                width=card_w - Inches(0.3), height=Inches(0.65),
                text=title, font_size=16, bold=True,
                color=RGB['white'], align=PP_ALIGN.CENTER)

    # 說明
    add_textbox(slide2,
                left=cx + Inches(0.2), top=start_y + Inches(1.55),
                width=card_w - Inches(0.4), height=Inches(3.0),
                text=body, font_size=13,
                color=RGB['white'], align=PP_ALIGN.LEFT)

# 底部結語
add_textbox(slide2,
            left=Inches(0.4), top=Inches(6.5),
            width=Inches(12.5), height=Inches(0.65),
            text='▶  解決方案：在上線流程中加入「設備訓練清單」審核機制，搭配自動化標註形成閉環。',
            font_size=14, bold=True, color=RGB['dark_blue'])


# ══════════════════════════════════════════════════════════════════════
# Slide 3 — 閉環流程圖（主要投影片）
# ══════════════════════════════════════════════════════════════════════

slide3 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide3, RGB['bg'])

# 頁首條
add_rect(slide3,
         left=Inches(0), top=Inches(0),
         width=SLIDE_W, height=Inches(0.75),
         fill_rgb=RGB['dark_blue'])
add_textbox(slide3,
            left=Inches(0.4), top=Inches(0.05),
            width=Inches(12), height=Inches(0.65),
            text='模型 / 算法優化  閉環流程圖',
            font_size=22, bold=True, color=RGB['white'])

# 嵌入流程圖圖片
pic = slide3.shapes.add_picture(
    FLOWCHART_IMG,
    left=Inches(0.15), top=Inches(0.85),
    height=Inches(6.45))


# ══════════════════════════════════════════════════════════════════════
# Slide 4 — 兩大觸發路徑說明
# ══════════════════════════════════════════════════════════════════════

slide4 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide4, RGB['bg'])

# 頁首條
add_rect(slide4,
         left=Inches(0), top=Inches(0),
         width=SLIDE_W, height=Inches(0.75),
         fill_rgb=RGB['dark_blue'])
add_textbox(slide4,
            left=Inches(0.4), top=Inches(0.05),
            width=Inches(12), height=Inches(0.65),
            text='閉環觸發路徑詳解',
            font_size=22, bold=True, color=RGB['white'])

# 路徑 A
add_rect(slide4,
         left=Inches(0.4), top=Inches(0.95),
         width=Inches(5.9), height=Inches(5.9),
         fill_rgb=RGB['dark_blue'])

add_textbox(slide4,
            left=Inches(0.55), top=Inches(1.05),
            width=Inches(5.6), height=Inches(0.55),
            text='路徑 A  ▸  設備不在訓練清單',
            font_size=17, bold=True, color=RGB['light_blue'])

path_a_steps = [
    ('①', '設備上拋圖像，系統查詢訓練清單'),
    ('②', '清單內無此設備型號 → 觸發自動化標註管線'),
    ('③', '自動標註完成 → 重新訓練模型'),
    ('④', '開發人員 Review 模型效能指標'),
    ('⑤', '確認通過 → 更新線上推論模型'),
    ('⑥', '新設備型號加入訓練清單，後續自動適配'),
]
add_bullet_slide_content(
    slide4, path_a_steps,
    left=Inches(0.55), top=Inches(1.7),
    width=Inches(5.6), height=Inches(4.8),
    font_size=13, color=RGB['white'], spacing_before=10)

# 路徑 B
add_rect(slide4,
         left=Inches(7.0), top=Inches(0.95),
         width=Inches(5.9), height=Inches(5.9),
         fill_rgb=RGB['steel_blue'])

add_textbox(slide4,
            left=Inches(7.15), top=Inches(1.05),
            width=Inches(5.6), height=Inches(0.55),
            text='路徑 B  ▸  在清單內但圖像質量偏移',
            font_size=17, bold=True, color=RGB['white'])

path_b_steps = [
    ('①', '設備在清單內，計算圖像質量指標（亮度/對比/清晰度等）'),
    ('②', '與基準值比較，差異超過閾值 → 判定為偏移'),
    ('③', '自動化標註新圖像，融入訓練集擴增泛化性'),
    ('④', '開發人員確認模型在新分布下效能'),
    ('⑤', '確認通過 → 推送更新後的模型'),
    ('⑥', '更新設備的圖像基準資料，防止未來誤觸'),
]
add_bullet_slide_content(
    slide4, path_b_steps,
    left=Inches(7.15), top=Inches(1.7),
    width=Inches(5.6), height=Inches(4.8),
    font_size=13, color=RGB['white'], spacing_before=10)

# 底部
add_textbox(slide4,
            left=Inches(0.4), top=Inches(7.05),
            width=Inches(12.5), height=Inches(0.35),
            text='兩條路徑均需開發人員最終確認，確保模型品質可控，再進行推論部署。',
            font_size=12, italic=True, color=RGB['gray'])


# ══════════════════════════════════════════════════════════════════════
# Slide 5 — 效益與關鍵特點
# ══════════════════════════════════════════════════════════════════════

slide5 = prs.slides.add_slide(blank_layout)
set_slide_bg(slide5, RGB['bg'])

# 頁首條
add_rect(slide5,
         left=Inches(0), top=Inches(0),
         width=SLIDE_W, height=Inches(0.75),
         fill_rgb=RGB['dark_blue'])
add_textbox(slide5,
            left=Inches(0.4), top=Inches(0.05),
            width=Inches(12), height=Inches(0.65),
            text='閉環流程  效益與關鍵特點',
            font_size=22, bold=True, color=RGB['white'])

benefits = [
    ('⚡', '快速響應',
     '新設備上線後自動觸發標註流程，無需等待用戶反饋，大幅縮短模型適應週期。',
     RGB['orange']),
    ('🔄', '持續自我優化',
     '每次設備更新或圖像偏移都會觸發訓練，模型隨業務成長持續累積泛化能力。',
     RGB['green']),
    ('🛡️', '人工品質把關',
     '自動化流程完成後，仍需開發人員確認指標，兼顧效率與模型安全性。',
     RGB['purple']),
    ('📉', '系統性降低過殺',
     '閉環設計讓過殺問題在上線初期即被攔截，而非依賴大量生產後的回退修復。',
     RGB['steel_blue']),
]

col_w = Inches(3.0)
col_h = Inches(5.1)
gap   = Inches(0.32)
bx    = Inches(0.42)
by    = Inches(1.0)

for i, (icon, title, body, clr) in enumerate(benefits):
    cx = bx + i * (col_w + gap)

    add_rect(slide5, cx, by, col_w, col_h, clr)

    add_textbox(slide5,
                left=cx + Inches(0.1), top=by + Inches(0.15),
                width=col_w - Inches(0.2), height=Inches(0.65),
                text=icon, font_size=32,
                color=RGB['white'], align=PP_ALIGN.CENTER)

    add_textbox(slide5,
                left=cx + Inches(0.12), top=by + Inches(0.85),
                width=col_w - Inches(0.24), height=Inches(0.65),
                text=title, font_size=16, bold=True,
                color=RGB['white'], align=PP_ALIGN.CENTER)

    add_textbox(slide5,
                left=cx + Inches(0.15), top=by + Inches(1.55),
                width=col_w - Inches(0.3), height=Inches(3.3),
                text=body, font_size=13,
                color=RGB['white'], align=PP_ALIGN.LEFT)

# 底部結語
add_rect(slide5,
         left=Inches(0.42), top=Inches(6.35),
         width=Inches(12.5), height=Inches(0.75),
         fill_rgb=RGB['dark_blue'])
add_textbox(slide5,
            left=Inches(0.55), top=Inches(6.4),
            width=Inches(12.2), height=Inches(0.65),
            text='目標：透過閉環機制，將「過殺問題」從被動修復轉變為主動預防，實現模型品質的持續提升。',
            font_size=14, bold=True,
            color=RGB['light_green'], align=PP_ALIGN.CENTER)


# ══════════════════════════════════════════════════════════════════════
# 存檔
# ══════════════════════════════════════════════════════════════════════

prs.save(OUTPUT_PPT)
print(f"✅ PPT 已儲存：{OUTPUT_PPT}")
print()
print("簡報包含以下投影片：")
print("  Slide 1 — 封面")
print("  Slide 2 — 問題背景（三大痛點）")
print("  Slide 3 — 閉環流程圖（主圖）")
print("  Slide 4 — 兩大觸發路徑詳解")
print("  Slide 5 — 效益與關鍵特點")
