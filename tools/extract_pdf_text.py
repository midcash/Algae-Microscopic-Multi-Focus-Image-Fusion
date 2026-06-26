import argparse
from pathlib import Path
from PyPDF2 import PdfReader


def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> str:
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    use_pages = total_pages if max_pages is None else min(max_pages, total_pages)

    parts: list[str] = []
    parts.append(f"FILE: {pdf_path.name}")
    parts.append(f"TOTAL_PAGES: {total_pages}")
    parts.append("")

    for i in range(use_pages):
        text = reader.pages[i].extract_text() or ""
        parts.append(f"=== PAGE {i + 1} ===")
        parts.append(text.strip())
        parts.append("")

    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF files using PyPDF2")
    parser.add_argument("pdfs", nargs="+", help="PDF file paths")
    parser.add_argument("--output-dir", default="docs/pdf_extracts", help="Directory to save extracted txt files")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional max pages to extract per PDF")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf in args.pdfs:
        pdf_path = Path(pdf)
        if not pdf_path.exists():
            print(f"[SKIP] Not found: {pdf_path}")
            continue

        try:
            text = extract_pdf_text(pdf_path, max_pages=args.max_pages)
            out_path = output_dir / f"{pdf_path.stem}.txt"
            out_path.write_text(text, encoding="utf-8")
            print(f"[OK] {pdf_path} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {pdf_path}: {e}")


if __name__ == "__main__":
    main()
