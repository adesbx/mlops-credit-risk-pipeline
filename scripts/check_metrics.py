import subprocess
import json
import sys


def get_dvc_metrics_diff():
    result = subprocess.run(
        ["dvc", "metrics", "diff", "--json"],
        capture_output=True,
        text=True,
        check=True
    )
    
    return json.loads(result.stdout)

metrics_data = get_dvc_metrics_diff()
accuracy_diff = metrics_data["models\\metrics.json"]["accuracy"]["diff"]
recall_diff = metrics_data["models\\metrics.json"]["recall"]["diff"]

if accuracy_diff < 0 or recall_diff < 0:
    print("Accuracy or recall diff is < 0")
    sys.exit(1)

else:
    print("Metrics ok")
    sys.exit(0)
