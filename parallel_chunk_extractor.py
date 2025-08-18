#!/usr/bin/env python3
"""
Parallel processing script to run chunk_extractor.py over multiple parquet files.
Processes all parquet files in the specified directory with configurable parallelism.
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional


def run_chunk_extractor(
    input_file: str,
    extract_infobox: bool = False,
    extract_link_graph: bool = False,
    page_meta_db: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Tuple[str, bool, str]:
    """
    Run chunk_extractor.py on a single parquet file.

    Args:
        input_file: Path to the parquet file
        extract_infobox: Whether to extract infobox data
        output_dir: Directory where output files should be written

    Returns:
        Tuple of (filename, success, error_message)
    """
    try:
        cmd = ["python", "chunk_extractor.py", "--input", input_file]

        if extract_infobox:
            cmd.append("--extract-infobox")
        if extract_link_graph:
            cmd.append("--extract-link-graph")
            if page_meta_db is not None:
                cmd.extend(["--page-meta-db", page_meta_db])

        # Set working directory - use output_dir if specified, otherwise script directory
        if output_dir and (extract_infobox or extract_link_graph):
            working_dir = output_dir
            os.makedirs(working_dir, exist_ok=True)
        else:
            working_dir = os.path.dirname(os.path.abspath(__file__))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3000,
            cwd=working_dir,
        )

        if result.returncode == 0:
            return (input_file, True, "")
        else:
            return (
                input_file,
                False,
                f"Exit code {result.returncode}: {result.stderr}",
            )

    except subprocess.TimeoutExpired:
        return (input_file, False, "Timeout after 5 minutes")
    except Exception as e:
        return (input_file, False, str(e))


def find_parquet_files(directory: str) -> List[str]:
    """
    Find all parquet files matching the wikipedia_en_part_*.parquet pattern.

    Args:
        directory: Directory to search for parquet files

    Returns:
        List of absolute paths to parquet files
    """
    parquet_dir = Path(directory)
    if not parquet_dir.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all files matching the pattern
    pattern = "wikipedia_en_part_*.parquet"
    parquet_files = list(parquet_dir.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found matching pattern {pattern} in {directory}"
        )

    # Sort files numerically by part number
    def extract_part_number(filepath):
        stem = filepath.stem  # wikipedia_en_part_001
        parts = stem.split("_")
        try:
            return int(parts[-1])  # Extract the numeric part
        except (ValueError, IndexError):
            return 0

    parquet_files.sort(key=extract_part_number)

    return [str(f.absolute()) for f in parquet_files]


def main():
    """Main function to process all parquet files in parallel"""
    parser = argparse.ArgumentParser(
        description="Run chunk_extractor.py in parallel over multiple parquet files"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="../wiki-extract/parquet",
        help="Directory containing parquet files (default: ../wiki-extract/parquet)",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=8,
        help="Number of parallel processes (default: 8)",
    )
    parser.add_argument(
        "--extract-infobox",
        action="store_true",
        help="Extract infobox data from articles",
    )
    parser.add_argument(
        "--extract-link-graph",
        action="store_true",
        help="Extract link graph from articles and store as docids in DuckDB",
    )
    parser.add_argument(
        "--page-meta-db",
        type=str,
        default=None,
        help="DuckDB file containing page_meta table for docid lookup",
    )
    parser.add_argument(
        "--max-files", type=int, help="Maximum number of files to process (for testing)"
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start processing from this part number (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually running",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where DuckDB files will be written (default: current directory)",
    )

    args = parser.parse_args()

    try:
        # Find all parquet files
        print(f"Searching for parquet files in: {args.input_dir}")
        all_files = find_parquet_files(args.input_dir)

        # Filter files based on start_from
        if args.start_from > 1:
            filtered_files = []
            for file_path in all_files:
                filename = Path(file_path).stem
                parts = filename.split("_")
                try:
                    part_num = int(parts[-1])
                    if part_num >= args.start_from:
                        filtered_files.append(file_path)
                except (ValueError, IndexError):
                    continue
            all_files = filtered_files

        # Limit files if max_files is specified
        if args.max_files:
            all_files = all_files[: args.max_files]

        print(f"Found {len(all_files)} parquet files to process")
        print(f"Parallelism: {args.parallelism}")
        print(f"Extract infobox: {args.extract_infobox}")
        print(f"Extract link graph: {args.extract_link_graph}")

        # Show output information
        if args.extract_infobox or args.extract_link_graph:
            output_dir = args.output_dir if args.output_dir else os.getcwd()
            print(f"DuckDB files will be written to: {output_dir}")
            if args.extract_infobox:
                print(f"DuckDB filename pattern: <parquet_filename>_infobox.duckdb")
            if args.extract_link_graph:
                print(f"DuckDB filename pattern: <parquet_filename>_linkgraph.duckdb")

        if args.dry_run:
            print("\nDry run - files that would be processed:")
            for i, file_path in enumerate(all_files[:10]):  # Show first 10
                filename = Path(file_path).name
                print(f"  {i+1}. {filename}")
                if args.extract_infobox:
                    db_name = f"{Path(file_path).stem}_infobox.duckdb"
                    print(f"      -> {db_name}")
            if len(all_files) > 10:
                print(f"  ... and {len(all_files) - 10} more files")
            return

        if not all_files:
            print("No files to process!")
            return

        # Process files in parallel
        start_time = time.time()
        successful = 0
        failed = 0

        print(f"\nStarting parallel processing...")
        print("=" * 80)

        with ProcessPoolExecutor(max_workers=args.parallelism) as executor:
            future_to_file = {
                executor.submit(
                    run_chunk_extractor,
                    file_path,
                    args.extract_infobox,
                    args.extract_link_graph,
                    args.page_meta_db,
                    args.output_dir,
                ): file_path
                for file_path in all_files
            }

            # Process completed jobs
            for future in as_completed(future_to_file):
                file_path, success, error_msg = future.result()
                filename = Path(file_path).name

                if success:
                    successful += 1
                    print(f"✓ [{successful + failed:4d}/{len(all_files)}] {filename}")
                else:
                    failed += 1
                    print(
                        f"✗ [{successful + failed:4d}/{len(all_files)}] {filename} - {error_msg}"
                    )

                # Show progress every 50 files
                if (successful + failed) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (successful + failed) / elapsed
                    remaining = len(all_files) - (successful + failed)
                    eta = remaining / rate if rate > 0 else 0
                    print(
                        f"   Progress: {successful + failed}/{len(all_files)} files, "
                        f"{rate:.1f} files/sec, ETA: {eta/60:.1f} minutes"
                    )

        # Final summary
        elapsed_time = time.time() - start_time
        print("=" * 80)
        print(f"Processing complete!")
        print(f"Total files: {len(all_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Average rate: {len(all_files)/elapsed_time:.1f} files/second")

        if failed > 0:
            print(
                f"\n{failed} files failed to process. Check the error messages above."
            )
            sys.exit(1)
        else:
            print("\nAll files processed successfully!")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
