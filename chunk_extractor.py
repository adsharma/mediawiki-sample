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


def process_article(docid, title, text, chunk_size=512):
    """Process a single article and print its chunks"""
    print(f"\n{'='*60}")
    print(f"Processing Article: {title}")
    print(f"Document ID: {docid}")
    print(f"{'='*60}")

    # Parse MediaWiki markup and strip formatting
    parsed_text = mwparserfromhell.parse(text).strip_code().strip()

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
        required=True,
        help="Process only the article with this specific document ID",
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
        query = f"""
            SELECT page_id, title, text 
            FROM read_parquet('{args.input}') 
            WHERE page_id = {args.docid} 
            AND NOT starts_with(text, '#REDIRECT')
        """

        result = conn.execute(query).fetchall()

        if not result:
            print(
                f"No article found with document ID {args.docid} (or it's a redirect page)"
            )
            return

        # Process the found article
        for row in result:
            docid, title, text = row
            print(f"Found article: {title}")

            # Process the article
            chunks = process_article(docid, title, text, args.chunk_size)
            print(f"Successfully processed document ID {args.docid}")

        conn.close()

    except FileNotFoundError:
        print(f"Error: Could not find parquet file: {args.input}")
    except Exception as e:
        print(f"Error processing parquet file: {e}")


if __name__ == "__main__":
    main()
