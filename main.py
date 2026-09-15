"""Command-line launcher for the canonical FitMotion AI processors."""
from curl import runcurl
from pushup import runpushup
from squat import runsquat


def main():
    actions = {
        "1": ("Squat", runsquat),
        "2": ("Hít đất", runpushup),
        "3": ("Bicep Curl tay trái", lambda: runcurl("left")),
        "4": ("Bicep Curl tay phải", lambda: runcurl("right")),
    }
    while True:
        print("\n===== FITMOTION AI - PIPELINE TUẦN 6 =====")
        for key, (label, _handler) in actions.items():
            print(f"{key}. {label}")
        print("0. Thoát")
        choice = input("Chọn bài tập: ").strip()
        if choice == "0":
            break
        item = actions.get(choice)
        if item is None:
            print("Lựa chọn không hợp lệ.")
            continue
        try:
            item[1]()
        except (RuntimeError, ValueError) as exc:
            print(f"Không thể chạy bài tập: {exc}")


if __name__ == "__main__":
    main()
