import json
from pathlib import Path
from datetime import datetime

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ========== 请在这里直接指定文件路径 ==========
GT_JSON = "../.././DATASETS/COCO2017/annotations/instances_val2017.json"        # COCO格式 ground truth
PRED_JSON = "./../improve1/runs/detect/train2/predictions.json"  # YOLO模型输出的预测json
OUT_JSON = "./runs/detect/train/predictions_coco_eval_ready.json"      # 输出json（会写入来源信息）

# ========== 推理参数（写入输出文件用于追溯） ==========
OUTPUT_CONFIDENCE_THRESHOLD = 0.001
OUTPUT_NMS_THRESHOLD = 0.7
OUTPUT_MAX_DETECTIONS = 300
OUTPUT_MAX_DETECTIONS_PER_CLASS = 300

COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]

def remap_and_filter_predictions(preds, conf_thres=0.001):
    out = []
    dropped_invalid_box = 0
    dropped_invalid_cls = 0
    dropped_low_conf = 0

    for d in preds:
        x, y, w, h = d["bbox"]
        score = float(d.get("score", 0.0))
        cid = int(d["category_id"])

        if w <= 0 or h <= 0:
            dropped_invalid_box += 1
            continue
        if score < conf_thres:
            dropped_low_conf += 1
            continue
        if not (1 <= cid <= 80):
            dropped_invalid_cls += 1
            continue

        out.append({
            "image_id": d["image_id"],
            "category_id": COCO80_TO_COCO91[cid - 1],
            "bbox": [float(x), float(y), float(w), float(h)],
            "score": score,
        })

    return out, {
        "kept": len(out),
        "dropped_invalid_box": dropped_invalid_box,
        "dropped_invalid_cls": dropped_invalid_cls,
        "dropped_low_conf": dropped_low_conf,
    }

def evaluate_coco_with_list(coco_gt, det_list):
    coco_dt = coco_gt.loadRes(det_list)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev.stats

def main():
    gt_path = str(Path(GT_JSON).resolve())
    pred_path = str(Path(PRED_JSON).resolve())
    out_path = str(Path(OUT_JSON).resolve())

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    remapped, proc_stats = remap_and_filter_predictions(
        preds, conf_thres=OUTPUT_CONFIDENCE_THRESHOLD
    )

    coco_gt = COCO(gt_path)
    stats = evaluate_coco_with_list(coco_gt, remapped)

    result = {
        "meta": {
            "created_at_utc": datetime.utcnow().isoformat() + "Z",
            "source_files": {  # 文件来源写在输出文件内部
                "ground_truth_json": gt_path,
                "prediction_json": pred_path,
                "output_json": out_path
            },
            "inference_settings": {
                "output_confidence_threshold": OUTPUT_CONFIDENCE_THRESHOLD,
                "output_nms_threshold": OUTPUT_NMS_THRESHOLD,
                "output_max_detections": OUTPUT_MAX_DETECTIONS,
                "output_max_detections_per_class": OUTPUT_MAX_DETECTIONS_PER_CLASS
            }
        },
        "preprocess_stats": proc_stats,
        "metrics": {
            "AP@[0.50:0.95]": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "APs": float(stats[3]),
            "APm": float(stats[4]),
            "APl": float(stats[5]),
            "AR1": float(stats[6]),
            "AR10": float(stats[7]),
            "AR100": float(stats[8])
        },
        "detections": remapped
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()