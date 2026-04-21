from pathlib import Path
import json

root = Path(".")
data_dir = root / "dataset"
qa = json.loads((data_dir / "qa.json").read_text(encoding="utf-8"))
images = sum(1 for p in (data_dir / "images").iterdir() if p.is_file())
pre = root / "preprocessed"

print("overall_qa:", len(qa))
print("overall_images:", images)
for name in ("qa_polar.json", "qa_numeric.json", "qa_descriptive.json"):
    path = pre / name
    rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    print(f"{path.stem}:", len(rows))
