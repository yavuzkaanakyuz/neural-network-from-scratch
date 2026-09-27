import argparse
import csv
import json
from pathlib import Path

from makemore_wavenet import (
    build_mlp,
    build_wavenet,
    count_parameters,
    evaluate,
    load_words,
    prepare_data,
    sample,
    smooth_loss,
    trace_shapes,
    train_model,
)


ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30000)
    args = parser.parse_args()

    english_words = load_words(ROOT / "names.txt")
    english3 = prepare_data(english_words, 3)
    english8 = prepare_data(english_words, 8)
    turkish_words = load_words(ROOT / "turkce_isim.csv")
    turkish3 = prepare_data(turkish_words, 3)
    turkish8 = prepare_data(turkish_words, 8)

    configs = [
        ("context 3 MLP", english3, lambda v: build_mlp(v, 3, 10, 200)),
        ("context 8 flat MLP", english8, lambda v: build_mlp(v, 8, 10, 200)),
        ("context 8 WaveNet wrong BN", english8, lambda v: build_wavenet(v, 10, 68, False)),
        ("context 8 WaveNet", english8, lambda v: build_wavenet(v, 10, 68, True)),
        ("context 8 WaveNet scaled", english8, lambda v: build_wavenet(v, 24, 128, True)),
        ("Turkish context 3 MLP", turkish3, lambda v: build_mlp(v, 3, 10, 200)),
        ("Turkish context 8 WaveNet", turkish8, lambda v: build_wavenet(v, 24, 128, True)),
    ]

    results, trained = [], {}
    for name, data, builder in configs:
        model = builder(len(data["stoi"]))
        history = train_model(model, data["train"], steps=args.steps)
        row = {
            "model": name,
            "parameters": count_parameters(model),
            "train_loss": evaluate(model, data["train"]),
            "dev_loss": evaluate(model, data["dev"]),
            "test_loss": evaluate(model, data["test"]),
            "steps": args.steps,
        }
        print(row, flush=True)
        results.append(row)
        trained[name] = (model, history, data)

    shape_model = build_wavenet(len(english8["stoi"]), 10, 68, True)
    shape_model.train()
    shapes = trace_shapes(shape_model, english8["train"][0][:4])
    generated = sample(
        trained["Turkish context 8 WaveNet"][0],
        turkish8["stoi"],
        turkish8["itos"],
    )
    loss_curves = {
        name: smooth_loss(history, group_size=1000).tolist()
        for name, (_, history, _) in trained.items()
    }
    payload = {
        "results": results,
        "shapes": shapes,
        "turkish_samples": generated,
        "loss_curves": loss_curves,
    }
    (ROOT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
