# print_test_acc.py
import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"\n=== {path} ===")
    print(f"avg test loss: {d['avg_test_loss']:.4f}")
    for task, m in d['per_task_test'].items():
        print(f"  {task.upper():6s}  loss: {m['loss']:.4f}  acc: {m['acc']*100:.2f}%")