#!/usr/bin/env python3
"""
Single CLI entry point to run either cloud classification method (profile-based
or cluster-based) for a date range, downloading any missing Cloudnet products
first, with the classification behaviour controlled by a JSON config file:

  * ``--method profile`` -> driven by a "hydrometeor sequence" config (which
    raw classification codes make up each cloud type, which codes may bridge
    a vertical gap, per-species gap thresholds, ...). See
    ``hydrometeor_configs/profile_hydrometeor_sequences.example.json``.
  * ``--method cluster`` -> driven by a "percentage of hydrometeors" config
    (min_liquid_percentage, min_ice_percentage, ...). See
    ``hydrometeor_configs/cluster_hydrometeor_percentages.example.json``.

Both methods are implemented in their own file (``profile_based_classification_cli.py``,
``cluster_classification_cli.py``) -- clean copies of the original research code,
not edits of it. This file only handles date-range/CLI/download orchestration
and dispatches to one of the two.

Usage
-----
    poetry run python phd_clouds/clouds/hydrometeor_classification_cli.py \\
        --method profile \\
        --start-date 2021-12-20 --end-date 2021-12-22 \\
        --site granada \\
        --data-dir /path/to/cloudnet_data \\
        --output-dir /path/to/processed_output \\
        --config phd_clouds/clouds/hydrometeor_configs/profile_hydrometeor_sequences.example.json

    poetry run python phd_clouds/clouds/hydrometeor_classification_cli.py \\
        --method cluster \\
        --start-date 2021-12-20 --end-date 2021-12-22 \\
        --data-dir /path/to/cloudnet_data \\
        --output-dir /path/to/processed_output \\
        --config phd_clouds/clouds/hydrometeor_configs/cluster_hydrometeor_percentages.example.json

Add ``--no-download`` if the Cloudnet files are already present under
``--data-dir`` (expected layout: ``<data-dir>/<product>/<date>_<site>_<product-file>.nc``,
one subfolder per product, e.g. ``<data-dir>/classification``, ``<data-dir>/categorize``, ...
-- exactly what ``--download`` itself produces).
"""
from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys

# `phd_clouds/clouds.py` (a module) and `phd_clouds/clouds/` (this directory)
# share a name and there is no `phd_clouds/clouds/__init__.py`, so this
# directory is NOT reachable as `phd_clouds.clouds.<this file>` -- Python
# resolves `phd_clouds.clouds` to the *module* `phd_clouds/clouds.py`.
# Sibling files in this same directory are therefore imported by bare name
# after adding this directory to sys.path, exactly like the two sibling CLI
# files below.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import profile_based_classification_cli as profile_cli  # noqa: E402
import cluster_classification_cli as cluster_cli  # noqa: E402
from phd_clouds.utils import download_cloudnet_products  # noqa: E402

logger = logging.getLogger("hydrometeor_classification_cli")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(THIS_DIR, "hydrometeor_configs")

# Cloudnet products needed per method, mapped to the subfolder each is
# downloaded into under --data-dir. Profile-based keeps lwc/iwc/der/ier in
# their own subfolders (profile_based_classification_cli.DayPaths has one
# field per product). Cluster-based's CloudProcessing.get_filenames_from_datastr
# looks up both lwc and iwc under a single `path_microphys` folder, so both
# are downloaded into a shared "microphys" subfolder for that method.
PRODUCTS_BY_METHOD = {
    "profile": {
        "classification": "classification", "categorize": "categorize", "radar": "radar",
        "lwc": "lwc", "iwc": "iwc", "der": "der", "ier": "ier",
    },
    "cluster": {
        "classification": "classification", "categorize": "categorize", "radar": "radar",
        "mwr": "mwr", "lwc": "microphys", "iwc": "microphys",
    },
}


def _parse_date(value: str) -> datetime.datetime:
    return datetime.datetime.strptime(value, "%Y-%m-%d")


def _date_range(start: datetime.datetime, end: datetime.datetime):
    days = (end.date() - start.date()).days
    return [start + datetime.timedelta(days=i) for i in range(days + 1)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run profile-based or cluster-based cloud classification over a date range.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", required=True, choices=["profile", "cluster"],
                         help="Classification method to run.")
    parser.add_argument("--start-date", required=True, type=_parse_date, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=_parse_date, default=None,
                         help="YYYY-MM-DD (default: same as --start-date)")
    parser.add_argument("--site", default="granada", help="Cloudnet site name (default: granada)")
    parser.add_argument("--data-dir", required=True,
                         help="Base directory for raw Cloudnet files (one subfolder per product).")
    parser.add_argument("--output-dir", default=None,
                         help="Where processed classification products are saved. "
                              "Required if --save-products is set.")
    parser.add_argument("--fig-dir", default=None, help="Where summary plots are saved (profile method).")
    parser.add_argument("--config", default=None,
                         help="Path to a hydrometeor-sequence (profile) or hydrometeor-percentage "
                              "(cluster) JSON config file. Defaults to the method's built-in defaults "
                              "(the same behaviour as the original scripts).")
    parser.add_argument("--download", dest="download", action="store_true", default=True,
                         help="Download missing Cloudnet products before processing (default).")
    parser.add_argument("--no-download", dest="download", action="store_false",
                         help="Skip downloading; assume files already exist under --data-dir.")
    parser.add_argument("--save-products", action="store_true",
                         help="Save classification outputs to --output-dir.")
    parser.add_argument("--make-plots", action="store_true",
                         help="Generate and save summary plots.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print what would be downloaded/processed and exit.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable INFO-level logging.")
    return parser


def download_missing_products(method: str, site: str, start: datetime.datetime, end: datetime.datetime,
                               data_dir: str) -> None:
    date_from = start.strftime("%Y-%m-%d")
    date_to = end.strftime("%Y-%m-%d")
    for product, subfolder in PRODUCTS_BY_METHOD[method].items():
        out_folder = os.path.join(data_dir, subfolder)
        logger.info("Downloading '%s' for %s..%s (site=%s) into %s", product, date_from, date_to, site, out_folder)
        download_cloudnet_products(date_from, date_to, out_folder, product=product, site=site)


def run_profile(args, dates) -> list:
    config = profile_cli.HydrometeorSequenceConfig.load(args.config)
    paths = profile_cli.DayPaths(
        classification=os.path.join(args.data_dir, "classification"),
        categorize=os.path.join(args.data_dir, "categorize"),
        radar=os.path.join(args.data_dir, "radar"),
        lwc=os.path.join(args.data_dir, "lwc"),
        iwc=os.path.join(args.data_dir, "iwc"),
        der=os.path.join(args.data_dir, "der"),
        ier=os.path.join(args.data_dir, "ier"),
    )
    options = profile_cli.RunOptions(
        output_dir=args.output_dir,
        fig_dir=args.fig_dir,
        save_products=args.save_products,
        make_plots=args.make_plots,
    )
    return profile_cli.run(dates, paths, config, options, site=args.site)


def run_cluster(args, dates) -> list:
    config = cluster_cli.ClusterConfig.load(args.config)
    paths = cluster_cli.DayPaths(
        classification=os.path.join(args.data_dir, "classification"),
        categorize=os.path.join(args.data_dir, "categorize"),
        radar=os.path.join(args.data_dir, "radar"),
        mwr=os.path.join(args.data_dir, "mwr"),
        microphys=os.path.join(args.data_dir, "microphys"),  # lwc + iwc together, see PRODUCTS_BY_METHOD
    )
    options = cluster_cli.RunOptions(
        output_dir=args.output_dir,
        save_products=args.save_products,
        make_plots=args.make_plots,
    )
    date_list = [d.strftime("%Y%m%d") for d in dates]
    return cluster_cli.run(date_list, args.site, paths, config, options)


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.end_date is None:
        args.end_date = args.start_date
    if args.end_date < args.start_date:
        parser.error("--end-date must not be before --start-date")
    if args.save_products and not args.output_dir:
        parser.error("--save-products requires --output-dir")

    dates = _date_range(args.start_date, args.end_date)

    if args.dry_run:
        print(f"Method: {args.method}")
        print(f"Site: {args.site}")
        print(f"Dates: {dates[0].strftime('%Y-%m-%d')}..{dates[-1].strftime('%Y-%m-%d')} ({len(dates)} day(s))")
        print(f"Config: {args.config or '(built-in default)'}")
        print(f"Would download products: {PRODUCTS_BY_METHOD[args.method]}" if args.download
              else "Download skipped (--no-download)")
        return 0

    if args.download:
        download_missing_products(args.method, args.site, args.start_date, args.end_date, args.data_dir)

    if args.method == "profile":
        results = run_profile(args, dates)
    else:
        results = run_cluster(args, dates)

    ok = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n{len(ok)}/{len(results)} day(s) processed successfully.")
    for r in failed:
        print(f"  FAILED {r['date']}: {r['error']}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
