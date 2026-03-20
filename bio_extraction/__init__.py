"""
bio_extraction
==============
Biographical entity extraction pipeline for Polish historical PDFs.

Processes documents through 6 sequential phases:
  1. Acquisition   — fetch PDFs from local disk or Common Crawl
  2. Classification — identify document type (directory, newspaper, civil record)
  3. Layout        — detect and slice entry regions
  4. OCR           — run Tesseract on each slice
  5. Extraction    — parse PersonEntity objects from OCR text
  6. Resolution    — deduplicate / merge entities into the SQLite database
"""
