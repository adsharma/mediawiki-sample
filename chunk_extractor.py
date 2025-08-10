#!/usr/bin/env python3
"""
Simplified script to extract and print chunks from Wikipedia articles in parquet files.
Contains only chunk_text() and a simplified version of process_article() for demonstration.
"""

import argparse
import mwparserfromhell
from tqdm import tqdm
import time
import duckdb
from pathlib import Path


def chunk_text(text, max_bytes):
    """Break text into chunks of maximum max_bytes size"""
    chunks = []
    text_bytes = text.encode("utf-8")

    if len(text_bytes) <= max_bytes:
        return [text]

    # Split by sentences first to try to maintain coherence
    sentences = text.split(". ")
    current_chunk = ""

    for sentence in sentences:
        sentence_with_period = (
            sentence + ". " if sentence != sentences[-1] else sentence
        )
        test_chunk = current_chunk + sentence_with_period

        if len(test_chunk.encode("utf-8")) <= max_bytes:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence_with_period
            else:
                # Single sentence is too long, split by words
                words = sentence.split()
                for word in words:
                    test_chunk = current_chunk + word + " "
                    if len(test_chunk.encode("utf-8")) <= max_bytes:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def extract_infobox(docid, title, text, input_filename):
    """Extract infobox data from a single article and store in DuckDB table"""
    # Create DuckDB filename based on input parquet filename
    input_path = Path(input_filename)
    db_filename = f"{input_path.stem}_infobox.duckdb"

    # Create DuckDB connection and table
    conn = duckdb.connect(str(db_filename))

    # Create table for storing infobox data
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS infobox_data (
            docid INTEGER,
            key VARCHAR,
            value VARCHAR
        )
    """
    )

    # Parse MediaWiki markup and strip formatting
    parsed_code = mwparserfromhell.parse(text)
    parsed_text = parsed_code.strip_code().strip()
    templates = parsed_code.filter_templates()

    for template in templates:
        if template.name.startswith("Infobox"):
            # Extract parameters from the infobox
            for param in template.params:
                param_name = param.name.strip()
                param_value = param.value.strip()
                # Remove wikitext markup for cleaner output (optional)
                param_value = mwparserfromhell.parse(param_value).strip_code()

                # Insert into DuckDB table
                conn.execute(
                    """
                    INSERT INTO infobox_data (docid, key, value)
                    VALUES (?, ?, ?)
                """,
                    [docid, param_name, param_value],
                )

    # Query and display the stored data
    result = conn.execute(
        """
        SELECT key, value FROM infobox_data
        WHERE docid = ?
        ORDER BY key
    """,
        [docid],
    ).fetchall()

    conn.close()


def process_article(docid, title, text, chunk_size=512):
    """Process a single article and print its chunks"""
    print(f"\n{'='*60}")
    print(f"Processing Article: {title}")
    print(f"Document ID: {docid}")
    print(f"{'='*60}")

    # Parse MediaWiki markup and strip formatting
    parsed_code = mwparserfromhell.parse(text)
    parsed_text = parsed_code.strip_code().strip()
    templates = parsed_code.filter_templates()
    for template in templates:
        if template.name.startswith("Infobox"):
            infobox_data = {}
            # Extract parameters from the infobox
            for param in template.params:
                param_name = param.name.strip()
                param_value = param.value.strip()
                # Remove wikitext markup for cleaner output (optional)
                param_value = mwparserfromhell.parse(param_value).strip_code()
                infobox_data[param_name] = param_value

            # Print the parsed data
            for key, value in infobox_data.items():
                print(f"{key}: {value}")

    # Generate chunks
    text_chunks = chunk_text(parsed_text, chunk_size)

    # Process chunks with progress bar showing kb/sec rate
    start_time = time.time()
    total_bytes_processed = 0

    print(f"Text length: {len(parsed_text)} characters")
    print(f"Number of chunks: {len(text_chunks)}")
    print(f"\nProcessing chunks...")

    with tqdm(text_chunks, desc="Processing chunks", unit="chunk") as pbar:
        for chunk_num, chunk in enumerate(pbar):
            chunk_bytes = len(chunk.encode("utf-8"))
            total_bytes_processed += chunk_bytes

            # Print chunk information
            print(f"\n--- Chunk {chunk_num + 1} ---")
            print(f"Size: {chunk_bytes} bytes")
            print(f"Content preview (first 512 chars):")
            print(chunk[:512] + ("..." if len(chunk) > 512 else ""))

            # Calculate and update processing rate
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                kb_per_sec = (total_bytes_processed / 1024) / elapsed_time
                pbar.set_postfix({"kb/sec": f"{kb_per_sec:.2f}", "bytes": chunk_bytes})

    print(f"\nSummary:")
    print(f"Total chunks processed: {len(text_chunks)}")
    print(
        f"Total bytes processed: {total_bytes_processed:,} bytes ({total_bytes_processed/1024:.2f} KB)"
    )

    return text_chunks


def main():
    """Main function to process articles from the parquet file"""
    parser = argparse.ArgumentParser(
        description="Extract and print chunks from Wikipedia articles in parquet files"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input parquet file to process"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Maximum chunk size in bytes (default: 512)",
    )
    parser.add_argument(
        "--docid",
        type=int,
        default=0,
        help="Process only the article with this specific document ID",
    )
    parser.add_argument(
        "--extract-infobox",
        action="store_true",
        help="Extract infobox data from the article",
    )

    args = parser.parse_args()

    print(f"Input parquet file: {args.input}")
    print(f"Chunk size: {args.chunk_size} bytes")
    print(f"Filtering for document ID: {args.docid}")

    try:
        # Use DuckDB to query the parquet file directly for the specific docid
        print(f"\nQuerying parquet file for document ID {args.docid}...")
        conn = duckdb.connect()

        # Query for the specific document ID, excluding redirects
        filter = f"page_id = '{args.docid}' AND" if args.docid else ""
        query = f"""
            SELECT page_id, title, text
            FROM read_parquet('{args.input}')
            WHERE
            {filter}
            NOT starts_with(text, '#REDIRECT')
        """

        result = conn.execute(query).df()

        if result.empty:
            print(
                f"No article found with document ID {args.docid} (or it's a redirect page)"
            )
            return

        # Process the found article
        for index, row in result.iterrows():
            docid = row["page_id"]
            title = row["title"]
            text = row["text"]
            print(f"Found article: {title}")

            # Process the article
            if args.extract_infobox:
                extract_infobox(docid, title, text, args.input)
            else:
                chunks = process_article(docid, title, text, args.chunk_size)
            print(f"Successfully processed document ID {docid}")

        conn.close()

    except FileNotFoundError:
        print(f"Error: Could not find parquet file: {args.input}")
    except Exception as e:
        print(f"Error processing parquet file: {e}")


if __name__ == "__main__":
    main()
