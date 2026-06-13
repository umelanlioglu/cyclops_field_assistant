from pathlib import Path
from ultralytics import YOLO


DATA_YAML = "data/demoday_train_yolo_aug3/dataset.yaml"
MODEL_NAME = "yolo26s-seg.pt"
RUN_NAME = "demoday_scene_overfit_aug3_no_mosaic"


def main():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"Missing {DATA_YAML}. Run scripts/prepare_coco_seg_demoday_aug3.py first."
        )

    model = YOLO(MODEL_NAME)

    model.train(
        task="segment",
        data=DATA_YAML,
        epochs=60,
        imgsz=640,
        batch=-1,
        device=0,
        workers=8,
        project="runs/printer_parts_seg",
        name=RUN_NAME,
        pretrained=True,
        plots=True,
        val=True,
        save=True,
        save_period=10,
        patience=0,
        optimizer="auto",
        seed=42,

        # Final Demo Day run used no mosaic-style training behavior.
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        cache="disk",
    )

    print("Best checkpoint:")
    print(f"runs/printer_parts_seg/{RUN_NAME}/weights/best.pt")


if __name__ == "__main__":
    main()
