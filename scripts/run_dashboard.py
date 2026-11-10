"""Open a single HTML view of saved graphs, metrics and segmentation examples."""

import argparse
import json
import sys
from pathlib import Path
import time

from samit.data.tiler import atomic_index
from samit.visualization.dashboard import build_dashboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/v2"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--watch", type=int, default=0, metavar="SECONDS")
    args = parser.parse_args()
    if args.watch < 0:
        parser.error("--watch must be nonnegative")
    output = args.output or args.results / "dashboard.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            document = build_dashboard(args.results, refresh_seconds=args.watch)
        except (FileNotFoundError, json.JSONDecodeError) as error:
            if not args.watch:
                raise
            print(f"Will retry reading results: {error}", file=sys.stderr, flush=True)
        else:
            with atomic_index(output) as stream:
                stream.write(document)
            print(f"Saved {output.resolve()}", flush=True)
        if not args.watch:
            return
        try:
            time.sleep(args.watch)
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
