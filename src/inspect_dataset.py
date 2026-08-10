from pathlib import Path

DATA_DIR = Path("data/pads-parkinsons-disease-smartwatch-dataset-1.0.0")

print("=" * 60)
print("PARKINSON'S PADS DATASET INSPECTION")
print("=" * 60)

if not DATA_DIR.exists():
    print(f"\nERROR: Dataset directory not found:")
    print(DATA_DIR.resolve())
    print("\nCheck that your dataset is located at:")
    print("parkinson-ai/data/pads-parkinsons-disease-smartwatch-dataset-1.0.0/")
    raise SystemExit(1)

print("\nDataset location:")
print(DATA_DIR.resolve())

print("\nTop-level contents:")
for item in DATA_DIR.iterdir():
    if item.is_dir():
        print(f"  [DIR]  {item.name}")
    else:
        print(f"  [FILE] {item.name}")

print("\nFile counts:")

extensions = {}

for path in DATA_DIR.rglob("*"):
    if path.is_file():
        suffix = path.suffix.lower() or "[no extension]"
        extensions[suffix] = extensions.get(suffix, 0) + 1

for extension, count in sorted(extensions.items()):
    print(f"  {extension:15} {count:,}")

print("\nInspection complete.")