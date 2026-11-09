"""C1 explicitly timed descriptive reports. Importing never writes files."""
import argparse
import json
from pathlib import Path
from report_metrics import SourceTimebase, build_report


def write_report(report, output_path):
    """Opt-in C1 JSON or Excel schema; never overwrite historical reports."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if output.suffix.lower() == '.json':
        payload = json.dumps(report, indent=2, allow_nan=False)
        with output.open('x', encoding='utf8') as f:
            f.write(payload+'\n')
    elif output.suffix.lower() == '.xlsx':
        import pandas as pd  # Optional; not needed for core metrics or JSON.
        with output.open('xb') as handle:
            with pd.ExcelWriter(handle, engine='openpyxl') as writer:
                pd.DataFrame([report['metadata']]).to_excel(writer, sheet_name='Metadata', index=False)
                for table, rows in report.items():
                    if table != 'metadata':
                        pd.DataFrame(rows).to_excel(writer, sheet_name=table[:31], index=False)
    else:
        raise ValueError("output must be a new .json or .xlsx path")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True, help='New output; existing files refused')
    parser.add_argument('--fps', type=float, help='Explicit nominal SOURCE FPS, not sampled FPS')
    parser.add_argument('--timestamps', help='Verified source-frame -> seconds JSON; takes priority over FPS')
    parser.add_argument('--frame-origin', type=int, default=0)
    parser.add_argument('--timestamp-origin', type=float, default=0.)
    parser.add_argument('--stride', type=int, required=True, help='Expected step in SOURCE frame indices')
    parser.add_argument('--coordinate-units', choices=['m','pixels','arbitrary'], required=True)
    parser.add_argument('--interval-seconds', type=float, default=30.)
    parser.add_argument('--max-gap-seconds', type=float, default=2.)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    timestamps = None
    if args.timestamps:
        with open(args.timestamps, encoding='utf8') as f:
            raw = json.load(f)
        if any(not k.isdigit() for k in raw):
            raise ValueError('timestamp keys must be nonnegative source-frame integers')
        timestamps = {int(k):v for k,v in raw.items()}
        if len(timestamps) != len(raw):
            raise ValueError('duplicate normalized timestamp frame keys')
    timebase = SourceTimebase(args.fps, timestamps, args.frame_origin, args.timestamp_origin)
    with open(args.input, encoding='utf8') as f:
        data = json.load(f)
    report = build_report(data, timebase, coordinate_units=args.coordinate_units,
                          expected_stride=args.stride, interval_seconds=args.interval_seconds,
                          max_gap_seconds=args.max_gap_seconds)
    write_report(report, args.output)


if __name__ == '__main__':
    main()
