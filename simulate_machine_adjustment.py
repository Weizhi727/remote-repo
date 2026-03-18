"""
simulate_machine_adjustment.py

模擬量產前機台調整後的圖像變化，用於測試 YOLO / GroundingDINO 的泛化性。

可模擬的變化：
  1. ROI 模糊（對焦偏移、鏡頭震動）
  2. 曝光變化（過曝 / 欠曝）
  3. 對比度變化
  4. 視野變化（縮放、平移、旋轉）
  5. 色溫偏移（白平衡失調）
  6. 複合擾動（以上隨機組合）

用法：
  # 單張圖片，套用所有單項模擬，輸出到 output/
  python simulate_machine_adjustment.py --input images/WS08_001.jpg --output output/

  # 批次處理資料夾
  python simulate_machine_adjustment.py --input images/ --output output/ --mode all

  # 只做複合擾動（最接近真實機台調整）
  python simulate_machine_adjustment.py --input images/ --output output/ --mode compound --n_compounds 5
"""

import argparse
import random
import json
from pathlib import Path

import cv2
import numpy as np


# ──────────────────────────────────────────────
# 1. 單項模擬函數
# ──────────────────────────────────────────────

def simulate_roi_blur(image: np.ndarray, intensity: float = 1.0) -> np.ndarray:
    """
    模擬對焦偏移或鏡頭輕微震動造成的 ROI 模糊。

    intensity: 0.0（輕微）~ 1.0（嚴重）
      - 0.0~0.3：輕微失焦，邊緣稍糊
      - 0.3~0.7：中度失焦，肉眼可見模糊
      - 0.7~1.0：嚴重失焦，焊點輪廓不清
    """
    # kernel size 對應模糊程度（必須為奇數）
    k = int(3 + intensity * 18)
    if k % 2 == 0:
        k += 1
    sigma = 1.0 + intensity * 5.0
    return cv2.GaussianBlur(image, (k, k), sigma)


def simulate_motion_blur(image: np.ndarray, intensity: float = 0.5, angle: float = 0.0) -> np.ndarray:
    """
    模擬傳送帶震動或機台晃動造成的運動模糊。

    intensity: 模糊長度（0~1 映射到 3~21 pixel）
    angle: 模糊方向（度，0=水平）
    """
    k = int(3 + intensity * 18)
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k

    # 旋轉 kernel 到指定角度
    M = cv2.getRotationMatrix2D((k // 2, k // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel = kernel / kernel.sum()

    return cv2.filter2D(image, -1, kernel)


def simulate_exposure(image: np.ndarray, ev_shift: float = 1.5) -> np.ndarray:
    """
    模擬曝光補償偏移（EV shift）。

    ev_shift > 0：過曝（圖像偏亮）
    ev_shift < 0：欠曝（圖像偏暗）
    建議範圍：-2.0 ~ +2.0 EV
    """
    factor = 2.0 ** ev_shift
    result = image.astype(np.float32) * factor
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_contrast(image: np.ndarray, alpha: float = 1.5, beta: float = 0.0) -> np.ndarray:
    """
    模擬對比度調整（例如光源老化、反光板污染）。

    alpha > 1：對比度增加
    alpha < 1：對比度降低（圖像偏灰）
    beta：亮度偏移（-50 ~ +50）
    """
    result = image.astype(np.float32) * alpha + beta
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_fov_change(
    image: np.ndarray,
    zoom: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    rotation: float = 0.0,
) -> np.ndarray:
    """
    模擬視野（FOV）變化：鏡頭重新定位、治具偏移。

    zoom:     縮放倍率（0.8=縮小視野=拉遠, 1.2=放大視野=拉近）
    tx, ty:   水平/垂直平移（比例，-0.1~0.1 代表 ±10% 圖寬/高）
    rotation: 旋轉角度（度，-5 ~ +5）
    """
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    # 建立仿射矩陣：先縮放/旋轉，再平移
    M_rotate_scale = cv2.getRotationMatrix2D((cx, cy), rotation, zoom)
    M_rotate_scale[0, 2] += tx * w
    M_rotate_scale[1, 2] += ty * h

    return cv2.warpAffine(
        image, M_rotate_scale, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,  # 邊界填充，避免黑邊
    )


def simulate_white_balance(image: np.ndarray, r_shift: float = 1.1, b_shift: float = 0.9) -> np.ndarray:
    """
    模擬白平衡偏移（色溫變化，例如燈管老化、換燈）。

    r_shift > 1：畫面偏暖（偏紅/橙）
    b_shift > 1：畫面偏冷（偏藍）
    建議範圍：0.8 ~ 1.3
    """
    result = image.astype(np.float32)
    result[:, :, 2] *= r_shift   # R channel（OpenCV BGR 格式，index=2）
    result[:, :, 0] *= b_shift   # B channel（index=0）
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_vignette(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    模擬鏡頭暗角（邊緣光量不足，常見於鏡頭組裝偏差）。

    strength: 0.0（無）~ 1.0（嚴重）
    """
    h, w = image.shape[:2]
    # 建立暗角 mask
    X = np.linspace(-1, 1, w)
    Y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(X, Y)
    radius = np.sqrt(xx**2 + yy**2)
    mask = 1.0 - np.clip(radius * strength, 0, 1)
    mask = mask[:, :, np.newaxis]  # 擴展到 3 通道

    result = image.astype(np.float32) * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_noise(image: np.ndarray, std: float = 15.0) -> np.ndarray:
    """
    模擬感光元件雜訊（例如換攝影機型號、高增益設定）。

    std: 雜訊標準差（5~30 為合理範圍）
    """
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────
# 2. 複合擾動
# ──────────────────────────────────────────────

# 各擾動的預設隨機範圍（對應「機台調整的合理幅度」）
PERTURBATION_RANGES = {
    "roi_blur": {
        "enabled": True,
        "intensity": (0.2, 0.8),         # 輕度到中度模糊
    },
    "motion_blur": {
        "enabled": True,
        "intensity": (0.1, 0.5),
        "angle": (0, 360),
    },
    "exposure": {
        "enabled": True,
        "ev_shift": (-1.5, 1.5),         # ±1.5 EV
    },
    "contrast": {
        "enabled": True,
        "alpha": (0.6, 1.6),             # 對比度壓縮或拉伸
        "beta": (-20, 20),
    },
    "fov": {
        "enabled": True,
        "zoom": (0.85, 1.15),            # ±15% 縮放
        "tx": (-0.08, 0.08),             # ±8% 平移
        "ty": (-0.08, 0.08),
        "rotation": (-3.0, 3.0),         # ±3 度旋轉
    },
    "white_balance": {
        "enabled": True,
        "r_shift": (0.85, 1.2),
        "b_shift": (0.85, 1.2),
    },
    "vignette": {
        "enabled": True,
        "strength": (0.2, 0.7),
    },
    "noise": {
        "enabled": True,
        "std": (5.0, 25.0),
    },
}


def apply_compound_perturbation(
    image: np.ndarray,
    config: dict,
    n_perturbations: int = 3,
    seed: int = None,
) -> tuple[np.ndarray, dict]:
    """
    從啟用的擾動中隨機選取 n_perturbations 種，依序套用。

    Returns:
        augmented_image: 擾動後的圖像
        applied_params: 本次套用的擾動參數記錄（用於可重現）
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 只從啟用的擾動中隨機選取
    available = [k for k, v in config.items() if v["enabled"]]
    selected = random.sample(available, min(n_perturbations, len(available)))

    result = image.copy()
    applied_params = {}

    for name in selected:
        cfg = config[name]

        if name == "roi_blur":
            intensity = random.uniform(*cfg["intensity"])
            result = simulate_roi_blur(result, intensity)
            applied_params[name] = {"intensity": round(intensity, 3)}

        elif name == "motion_blur":
            intensity = random.uniform(*cfg["intensity"])
            angle = random.uniform(*cfg["angle"])
            result = simulate_motion_blur(result, intensity, angle)
            applied_params[name] = {"intensity": round(intensity, 3), "angle": round(angle, 1)}

        elif name == "exposure":
            ev = random.uniform(*cfg["ev_shift"])
            result = simulate_exposure(result, ev)
            applied_params[name] = {"ev_shift": round(ev, 3)}

        elif name == "contrast":
            alpha = random.uniform(*cfg["alpha"])
            beta = random.uniform(*cfg["beta"])
            result = simulate_contrast(result, alpha, beta)
            applied_params[name] = {"alpha": round(alpha, 3), "beta": round(beta, 1)}

        elif name == "fov":
            zoom = random.uniform(*cfg["zoom"])
            tx = random.uniform(*cfg["tx"])
            ty = random.uniform(*cfg["ty"])
            rotation = random.uniform(*cfg["rotation"])
            result = simulate_fov_change(result, zoom, tx, ty, rotation)
            applied_params[name] = {
                "zoom": round(zoom, 3),
                "tx": round(tx, 4),
                "ty": round(ty, 4),
                "rotation": round(rotation, 2),
            }

        elif name == "white_balance":
            r_shift = random.uniform(*cfg["r_shift"])
            b_shift = random.uniform(*cfg["b_shift"])
            result = simulate_white_balance(result, r_shift, b_shift)
            applied_params[name] = {"r_shift": round(r_shift, 3), "b_shift": round(b_shift, 3)}

        elif name == "vignette":
            strength = random.uniform(*cfg["strength"])
            result = simulate_vignette(result, strength)
            applied_params[name] = {"strength": round(strength, 3)}

        elif name == "noise":
            std = random.uniform(*cfg["std"])
            result = simulate_noise(result, std)
            applied_params[name] = {"std": round(std, 2)}

    return result, applied_params


# ──────────────────────────────────────────────
# 3. 批次處理
# ──────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def collect_images(input_path: Path) -> list[Path]:
    """收集輸入路徑下的所有圖像檔案。"""
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def run_single_perturbations(
    images: list[Path],
    output_dir: Path,
    config: dict,
) -> None:
    """為每張圖像生成所有單項擾動的樣本（強度取各範圍中點）。"""
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] 無法讀取：{img_path}")
            continue

        stem = img_path.stem

        perturbations = {
            "blur_light":    lambda im: simulate_roi_blur(im, 0.2),
            "blur_heavy":    lambda im: simulate_roi_blur(im, 0.7),
            "overexpose":    lambda im: simulate_exposure(im, 1.2),
            "underexpose":   lambda im: simulate_exposure(im, -1.2),
            "low_contrast":  lambda im: simulate_contrast(im, 0.6, 0),
            "high_contrast": lambda im: simulate_contrast(im, 1.6, 0),
            "fov_zoomin":    lambda im: simulate_fov_change(im, zoom=1.15),
            "fov_zoomout":   lambda im: simulate_fov_change(im, zoom=0.85),
            "fov_shift":     lambda im: simulate_fov_change(im, tx=0.07, ty=0.05),
            "fov_rotate":    lambda im: simulate_fov_change(im, rotation=3.0),
            "wb_warm":       lambda im: simulate_white_balance(im, r_shift=1.2, b_shift=0.85),
            "wb_cool":       lambda im: simulate_white_balance(im, r_shift=0.85, b_shift=1.2),
            "vignette":      lambda im: simulate_vignette(im, 0.5),
            "noise":         lambda im: simulate_noise(im, 20.0),
            "motion_blur":   lambda im: simulate_motion_blur(im, 0.4, 0),
        }

        for name, fn in perturbations.items():
            out_subdir = output_dir / name
            out_subdir.mkdir(parents=True, exist_ok=True)
            result = fn(image)
            out_path = out_subdir / f"{stem}_{name}.jpg"
            cv2.imwrite(str(out_path), result)

        print(f"[OK] {img_path.name} → {len(perturbations)} 種單項擾動")


def run_compound_perturbations(
    images: list[Path],
    output_dir: Path,
    config: dict,
    n_compounds: int = 5,
    n_perturbations: int = 3,
    base_seed: int = 42,
) -> None:
    """
    為每張圖像生成 n_compounds 種複合擾動樣本，並記錄參數 JSON。
    """
    params_log = {}
    out_subdir = output_dir / "compound"
    out_subdir.mkdir(parents=True, exist_ok=True)

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARN] 無法讀取：{img_path}")
            continue

        stem = img_path.stem
        params_log[stem] = []

        for i in range(n_compounds):
            seed = base_seed + i * 1000 + hash(stem) % 1000
            result, applied = apply_compound_perturbation(
                image, config, n_perturbations=n_perturbations, seed=seed
            )
            out_name = f"{stem}_compound_{i:02d}.jpg"
            cv2.imwrite(str(out_subdir / out_name), result)
            params_log[stem].append({"variant": i, "seed": seed, "applied": applied})

        print(f"[OK] {img_path.name} → {n_compounds} 種複合擾動")

    # 儲存擾動參數記錄（便於重現）
    log_path = out_subdir / "perturbation_params.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(params_log, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] 擾動參數已記錄至：{log_path}")


# ──────────────────────────────────────────────
# 4. 主程式入口
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="模擬機台調整後的圖像變化，用於測試模型泛化性"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="輸入圖像路徑（單張圖片 或 資料夾）"
    )
    parser.add_argument(
        "--output", "-o", default="output_simulated",
        help="輸出資料夾路徑（預設：output_simulated）"
    )
    parser.add_argument(
        "--mode", choices=["all", "single", "compound"], default="all",
        help="模式：all=全部 | single=只做單項擾動 | compound=只做複合擾動（預設：all）"
    )
    parser.add_argument(
        "--n_compounds", type=int, default=5,
        help="每張圖片生成幾種複合擾動樣本（預設：5）"
    )
    parser.add_argument(
        "--n_perturbations", type=int, default=3,
        help="每次複合擾動套用幾種擾動（預設：3）"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="基礎隨機種子（預設：42，確保可重現）"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)
    if not images:
        print(f"[ERROR] 找不到圖像檔案：{input_path}")
        return

    print(f"[INFO] 共找到 {len(images)} 張圖像")
    print(f"[INFO] 輸出資料夾：{output_dir}")
    print(f"[INFO] 模式：{args.mode}\n")

    if args.mode in ("all", "single"):
        print("=== 單項擾動 ===")
        run_single_perturbations(images, output_dir, PERTURBATION_RANGES)

    if args.mode in ("all", "compound"):
        print("\n=== 複合擾動 ===")
        run_compound_perturbations(
            images, output_dir, PERTURBATION_RANGES,
            n_compounds=args.n_compounds,
            n_perturbations=args.n_perturbations,
            base_seed=args.seed,
        )

    print("\n[完成] 全部模擬圖像已輸出。")
    print(f"[INFO] 輸出結構：")
    print(f"  {output_dir}/")
    print(f"  ├── blur_light/         # 單項：輕度模糊")
    print(f"  ├── blur_heavy/         # 單項：重度模糊")
    print(f"  ├── overexpose/         # 單項：過曝")
    print(f"  ├── underexpose/        # 單項：欠曝")
    print(f"  ├── ... (共 15 種單項)")
    print(f"  └── compound/           # 複合擾動")
    print(f"      ├── *_compound_00.jpg")
    print(f"      ├── *_compound_01.jpg")
    print(f"      └── perturbation_params.json  ← 記錄每張圖的擾動參數")


if __name__ == "__main__":
    main()
