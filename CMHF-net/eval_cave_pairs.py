#!/usr/bin/env python3
"""Evaluate CMHF-net checkpoint on paired CAVE-style HSI/RGB MAT files."""

import argparse
import glob
import os
import sys
import types
from typing import List, Tuple

import numpy as np
import scipy.io as sio

CAVE_RATIO = 32
OUT_DIM = 31
UP_RANK = 12
HSINET_LAYERS = 20
SUBNET_LAYERS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CMHF-net eval on paired HSI/RGB MAT files")
    parser.add_argument("--mode", default="eval", choices=["eval"], help="Execution mode")
    parser.add_argument("--test_hsi_dir", required=True, help="Directory containing HSI .mat files")
    parser.add_argument("--test_rgb_dir", required=True, help="Directory containing RGB .mat files")
    parser.add_argument(
        "--weights_path",
        required=True,
        help="Checkpoint prefix (e.g., model-epoch-30) or checkpoint directory",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to save output MAT files")
    parser.add_argument(
        "--response_mat",
        default="rowData/CAVEdata/response coefficient.mat",
        help="Path to response coefficient .mat containing C (and optionally R)",
    )
    parser.add_argument("--gpu_ids", default="0", help="CUDA visible devices value, e.g. 0 or 0,1")
    parser.add_argument(
        "--rgb_source",
        default="provided",
        choices=["provided", "synth"],
        help="Use paired RGB MAT files (provided) or synthesize RGB from HSI using response R (synth)",
    )
    parser.add_argument(
        "--psnr_range",
        default="one",
        choices=["one", "dtype2", "gtmax"],
        help="PSNR data range: one=1.0, dtype2=2.0 (legacy float behavior), gtmax=max(gt)-min(gt)",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for .mat files")
    parser.add_argument(
        "--normalize_255",
        action="store_true",
        help="Force divide inputs by 255. If not set, auto-normalize if max > 1.5",
    )
    return parser.parse_args()


def resolve_checkpoint(weights_path: str, tf_mod) -> str:
    if os.path.isdir(weights_path):
        ckpt = tf_mod.train.latest_checkpoint(weights_path)
        if ckpt:
            return ckpt
        checkpoint_txt = os.path.join(weights_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            return weights_path
        raise FileNotFoundError(f"No checkpoint found in directory: {weights_path}")

    # If user passes .index/.meta, strip extension.
    for suffix in (".index", ".meta", ".data-00000-of-00001"):
        if weights_path.endswith(suffix):
            weights_path = weights_path[: -len(suffix)]
            break

    if os.path.exists(weights_path + ".index"):
        return weights_path

    raise FileNotFoundError(f"Checkpoint not found for weights_path: {weights_path}")


def load_response_c_and_r(response_mat: str) -> Tuple[np.ndarray, np.ndarray]:
    data = sio.loadmat(response_mat)
    c = None
    r = None
    if "C" in data and isinstance(data["C"], np.ndarray):
        c = data["C"]
    if "R" in data and isinstance(data["R"], np.ndarray):
        r = data["R"]

    if c is None:
        for key, value in data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.shape == (32, 32):
                c = value
                break

    if c is None or c.shape != (32, 32):
        raise KeyError(f"Could not find 32x32 C in {response_mat}")

    if r is not None:
        if r.shape == (31, 3):
            pass
        elif r.shape == (3, 31):
            r = r.T
        else:
            r = None

    return c.astype(np.float32), (None if r is None else r.astype(np.float32))


def list_mat_files(folder: str, recursive: bool) -> List[str]:
    pattern = "**/*.mat" if recursive else "*.mat"
    return sorted(glob.glob(os.path.join(folder, pattern), recursive=recursive))


def load_candidate_cube(mat_path: str, prefer_rgb: bool) -> Tuple[np.ndarray, str]:
    data = sio.loadmat(mat_path)
    candidates: List[Tuple[str, np.ndarray]] = []
    for key, value in data.items():
        if key.startswith("__"):
            continue
        if isinstance(value, np.ndarray) and value.ndim == 3:
            candidates.append((key, value))

    if not candidates:
        raise ValueError(f"No 3D array found in {mat_path}")

    if prefer_rgb:
        for key, value in candidates:
            if value.shape[2] == 3:
                return value.astype(np.float32), key
    else:
        for key, value in candidates:
            if value.shape[2] > 3:
                return value.astype(np.float32), key

    key, value = candidates[0]
    return value.astype(np.float32), key


def auto_normalize(x: np.ndarray, force_255: bool) -> np.ndarray:
    if force_255 or np.nanmax(x) > 1.5:
        return x / 255.0
    return x


def build_z_from_x(x_hsi: np.ndarray, c: np.ndarray, ratio: int = CAVE_RATIO) -> np.ndarray:
    h, w, bands = x_hsi.shape
    if h % ratio != 0 or w % ratio != 0:
        raise ValueError(f"Shape {x_hsi.shape} must be divisible by ratio={ratio}")
    z = np.zeros((h // ratio, w // ratio, bands), dtype=np.float32)
    for j in range(ratio):
        for k in range(ratio):
            z += x_hsi[j:h:ratio, k:w:ratio, :] * c[k, j]
    return z


def psnr(gt: np.ndarray, pred: np.ndarray, psnr_range: str = "one") -> float:
    mse = np.mean((gt - pred) ** 2)
    if mse <= 1e-12:
        return 99.0
    if psnr_range == "dtype2":
        peak = 2.0
    elif psnr_range == "gtmax":
        peak = max(float(np.max(gt) - np.min(gt)), 1e-12)
    else:
        peak = 1.0
    return float(10.0 * np.log10((peak * peak) / mse))


def sam_deg(gt: np.ndarray, pred: np.ndarray) -> float:
    gt2 = gt.reshape(-1, gt.shape[2])
    pr2 = pred.reshape(-1, pred.shape[2])
    dot = np.sum(gt2 * pr2, axis=1)
    ng = np.linalg.norm(gt2, axis=1)
    npred = np.linalg.norm(pr2, axis=1)
    denom = np.maximum(ng * npred, 1e-12)
    cosv = np.clip(dot / denom, -1.0, 1.0)
    return float(np.mean(np.degrees(np.arccos(cosv))))


def rmse(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((gt - pred) ** 2)))


def ergas(gt: np.ndarray, pred: np.ndarray, ratio: int = CAVE_RATIO) -> float:
    # ERGAS for HSI: mean over bands of relative RMSE, scaled by spatial ratio.
    err2 = (gt - pred) ** 2
    rmse_per_band = np.sqrt(np.mean(err2, axis=(0, 1)))
    mean_per_band = np.mean(gt, axis=(0, 1))
    denom = np.maximum(mean_per_band, 1e-12)
    val = (100.0 / ratio) * np.sqrt(np.mean((rmse_per_band / denom) ** 2))
    return float(val)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.makedirs(args.output_dir, exist_ok=True)

    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    # Minimal shim for tf.contrib usage inside original model code.
    if not hasattr(tf, "contrib"):
        def tf1_l2_regularizer(scale: float):
            # Match TF1 contrib behavior and support ref variables.
            def _regularizer(var):
                return tf.multiply(scale, tf.nn.l2_loss(var))

            return _regularizer

        tf.contrib = types.SimpleNamespace(
            layers=types.SimpleNamespace(l2_regularizer=tf1_l2_regularizer)
        )

    # Make local imports in this legacy codebase see tf.compat.v1 as tensorflow.
    sys.modules["tensorflow"] = tf
    import MHFnet  # pylint: disable=import-error,import-outside-toplevel

    ckpt = resolve_checkpoint(args.weights_path, tf)
    c, r = load_response_c_and_r(args.response_mat)

    hsi_files = list_mat_files(args.test_hsi_dir, args.recursive)
    if not hsi_files:
        raise FileNotFoundError(f"No .mat files found in {args.test_hsi_dir}")

    y = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    z = tf.placeholder(tf.float32, shape=[1, None, None, OUT_DIM])
    out_x, _, out_ya, _, out_hy = MHFnet.HSInet(
        y,
        z,
        iniUp3x3=0,
        iniA=0,
        upRank=UP_RANK,
        outDim=OUT_DIM,
        HSInetL=HSINET_LAYERS,
        subnetL=SUBNET_LAYERS,
        ratio=CAVE_RATIO,
    )

    saver = tf.train.Saver(max_to_keep=5)

    metrics_rows: List[Tuple[str, float, float, float, float]] = []

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, ckpt)

        done = 0
        for hsi_path in hsi_files:
            name = os.path.basename(hsi_path)
            rgb_path = os.path.join(args.test_rgb_dir, name)
            if args.rgb_source == "provided" and not os.path.exists(rgb_path):
                print(f"[Skip] RGB pair not found for {name}")
                continue

            x_hsi, hsi_key = load_candidate_cube(hsi_path, prefer_rgb=False)
            if args.rgb_source == "provided":
                y_rgb, rgb_key = load_candidate_cube(rgb_path, prefer_rgb=True)
            else:
                if r is None:
                    raise ValueError(
                        "rgb_source=synth requires response_mat to contain R with shape (31,3) or (3,31)"
                    )
                y_rgb = np.tensordot(x_hsi, r, axes=(2, 0)).astype(np.float32)
                rgb_key = "SYNTH_FROM_R"

            if x_hsi.shape[2] != OUT_DIM:
                raise ValueError(
                    f"{name}: expected {OUT_DIM} HSI bands, got {x_hsi.shape[2]} (key={hsi_key})"
                )
            if y_rgb.shape[2] != 3:
                raise ValueError(f"{name}: expected RGB 3 channels, got {y_rgb.shape[2]} (key={rgb_key})")
            if x_hsi.shape[:2] != y_rgb.shape[:2]:
                raise ValueError(
                    f"{name}: spatial mismatch HSI{x_hsi.shape[:2]} vs RGB{y_rgb.shape[:2]}"
                )

            x_hsi = auto_normalize(x_hsi, args.normalize_255)
            y_rgb = auto_normalize(y_rgb, args.normalize_255)

            if args.rgb_source == "provided" and r is not None:
                y_syn = np.tensordot(x_hsi, r, axes=(2, 0)).astype(np.float32)
                y_syn = auto_normalize(y_syn, args.normalize_255)
                rgb_gap_rmse = float(np.sqrt(np.mean((y_rgb - y_syn) ** 2)))
                print(f"[Info] {name} | provided-vs-synth RGB RMSE={rgb_gap_rmse:.6f}")

            z_msi = build_z_from_x(x_hsi, c, CAVE_RATIO)

            pred_x, pred_ya, pred_hy = sess.run(
                [out_x, out_ya, out_hy],
                feed_dict={
                    y: np.expand_dims(y_rgb, axis=0),
                    z: np.expand_dims(z_msi, axis=0),
                },
            )

            pred_x = np.squeeze(pred_x, axis=0)
            pred_ya = np.squeeze(pred_ya, axis=0)
            pred_hy = np.squeeze(pred_hy, axis=0)

            scene_psnr = psnr(x_hsi, pred_x, args.psnr_range)
            scene_sam = sam_deg(x_hsi, pred_x)
            scene_rmse = rmse(x_hsi, pred_x)
            scene_ergas = ergas(x_hsi, pred_x, CAVE_RATIO)
            metrics_rows.append((name, scene_psnr, scene_sam, scene_rmse, scene_ergas))

            out_path = os.path.join(args.output_dir, name)
            sio.savemat(
                out_path,
                {
                    "outX": pred_x,
                    "outYA": pred_ya,
                    "outHY": pred_hy,
                    "RGB_in": y_rgb,
                    "HSI_in": x_hsi,
                    "Zmsi_in": z_msi,
                    "PSNR": np.array([[scene_psnr]], dtype=np.float32),
                    "SAM_deg": np.array([[scene_sam]], dtype=np.float32),
                    "RMSE": np.array([[scene_rmse]], dtype=np.float32),
                    "ERGAS": np.array([[scene_ergas]], dtype=np.float32),
                },
            )
            done += 1
            print(
                f"[Done] {name} | "
                f"PSNR={scene_psnr:.4f}, SAM={scene_sam:.4f}, "
                f"RMSE={scene_rmse:.6f}, ERGAS={scene_ergas:.6f}"
            )

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("scene,psnr,sam_deg,rmse,ergas\n")
        for scene, p, s, r, e in metrics_rows:
            f.write(f"{scene},{p:.6f},{s:.6f},{r:.8f},{e:.8f}\n")
        if metrics_rows:
            mean_psnr = float(np.mean([x[1] for x in metrics_rows]))
            mean_sam = float(np.mean([x[2] for x in metrics_rows]))
            mean_rmse = float(np.mean([x[3] for x in metrics_rows]))
            mean_ergas = float(np.mean([x[4] for x in metrics_rows]))
            f.write(f"MEAN,{mean_psnr:.6f},{mean_sam:.6f},{mean_rmse:.8f},{mean_ergas:.8f}\n")

    if metrics_rows:
        mean_psnr = float(np.mean([x[1] for x in metrics_rows]))
        mean_sam = float(np.mean([x[2] for x in metrics_rows]))
        mean_rmse = float(np.mean([x[3] for x in metrics_rows]))
        mean_ergas = float(np.mean([x[4] for x in metrics_rows]))
        summary_path = os.path.join(args.output_dir, "metrics_summary.mat")
        sio.savemat(
            summary_path,
            {
                "mean_psnr": np.array([[mean_psnr]], dtype=np.float32),
                "mean_sam_deg": np.array([[mean_sam]], dtype=np.float32),
                "mean_rmse": np.array([[mean_rmse]], dtype=np.float32),
                "mean_ergas": np.array([[mean_ergas]], dtype=np.float32),
                "scene_names": np.array([x[0] for x in metrics_rows], dtype=object),
                "scene_psnr": np.array([[x[1]] for x in metrics_rows], dtype=np.float32),
                "scene_sam_deg": np.array([[x[2]] for x in metrics_rows], dtype=np.float32),
                "scene_rmse": np.array([[x[3]] for x in metrics_rows], dtype=np.float32),
                "scene_ergas": np.array([[x[4]] for x in metrics_rows], dtype=np.float32),
            },
        )

    print(f"Finished. Processed {len(metrics_rows)} scenes.")
    print(f"Saved predictions to: {args.output_dir}")
    print(f"Saved metrics to: {metrics_path}")
    if metrics_rows:
        print(
            "Average metrics | "
            f"PSNR={mean_psnr:.4f}, SAM={mean_sam:.4f}, "
            f"RMSE={mean_rmse:.6f}, ERGAS={mean_ergas:.6f}"
        )


if __name__ == "__main__":
    main()
