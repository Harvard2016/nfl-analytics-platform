"""Highlight commands: `bin/highlights-lens audit | run | export`."""
from __future__ import annotations

import json
import sys

from . import export, models, pipeline


def main() -> None:
    cmd = (sys.argv[1:] or ["run"])[0]
    if cmd == "audit":
        r = pipeline.audit()
        print(json.dumps({k: r[k] for k in ("videos", "leagues", "seasons_in_titles", "clips", "positive_share", "duplicate_source_links", "problems", "feature_dims")}, indent=1))
        print(json.dumps({k: pipeline.make_split()[k] for k in ("train", "validation", "test")}))
    elif cmd == "run":
        r = models.run()
        for n, v in r["results"].items():
            print(f"{n:48s} val mAP {v['validation']['summary']['mean_average_precision']:.4f}  test mAP {v['test']['summary']['mean_average_precision']:.4f}  "
                  f"test 3-min precision {v['test']['summary']['3 minutes']['precision']:.3f}")
        print("shipped:", r["shipped"])
    elif cmd == "export":
        print(export.run())
    else:
        raise SystemExit("usage: highlights-lens audit | run | export")


if __name__ == "__main__":
    main()
