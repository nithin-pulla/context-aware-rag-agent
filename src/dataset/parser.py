"""Document parsing utilities for various formats."""

import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from markdown import markdown
from pypdf import PdfReader

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentParser:
    """Parse technical documentation from various formats."""

    @staticmethod
    def parse_markdown(file_path: Path) -> Dict:
        """
        Parse Markdown file and extract structured content.

        Args:
            file_path: Path to .md file

        Returns:
            Dictionary with source, sections, and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                md_content = f.read()

            # Convert to HTML for structure extraction
            html = markdown(md_content, extensions=['extra', 'codehilite', 'tables'])
            soup = BeautifulSoup(html, 'html.parser')

            # Extract sections based on headings
            sections = []
            for heading in soup.find_all(['h1', 'h2', 'h3']):
                title = heading.get_text().strip()
                content = []

                # Collect content until next heading
                for sibling in heading.find_next_siblings():
                    if sibling.name in ['h1', 'h2', 'h3']:
                        break
                    content.append(sibling.get_text())

                sections.append({
                    'title': title,
                    'content': '\n'.join(content).strip(),
                    'level': int(heading.name[1])
                })

            # Compute content hash
            content_hash = hashlib.md5(md_content.encode()).hexdigest()

            return {
                'source': str(file_path),
                'sections': sections,
                'full_text': soup.get_text(separator='\n', strip=True),
                'metadata': {
                    'format': 'markdown',
                    'content_hash': content_hash,
                    'file_size': file_path.stat().st_size
                }
            }

        except Exception as e:
            logger.error(f"Error parsing markdown {file_path}: {e}")
            raise

    @staticmethod
    def parse_pdf(file_path: Path) -> Dict:
        """
        Parse PDF file and extract text content.

        Args:
            file_path: Path to .pdf file

        Returns:
            Dictionary with source, pages, and metadata
        """
        try:
            reader = PdfReader(str(file_path))
            pages = []

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                pages.append({
                    'page_num': page_num + 1,
                    'text': text.strip()
                })

            # Combine all pages
            full_text = '\n\n'.join(p['text'] for p in pages)
            content_hash = hashlib.md5(full_text.encode()).hexdigest()

            return {
                'source': str(file_path),
                'pages': pages,
                'full_text': full_text,
                'metadata': {
                    'format': 'pdf',
                    'total_pages': len(pages),
                    'content_hash': content_hash,
                    'file_size': file_path.stat().st_size
                }
            }

        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            raise

    @staticmethod
    def parse_html(file_path: Path) -> Dict:
        """
        Parse HTML file and extract main content.

        Args:
            file_path: Path to .html file

        Returns:
            Dictionary with source, text, and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

            # Remove script and style tags
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()

            # Try to find main content area
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                soup.find('body')
            )

            text = main_content.get_text(separator='\n', strip=True)
            content_hash = hashlib.md5(text.encode()).hexdigest()

            return {
                'source': str(file_path),
                'text': text,
                'full_text': text,
                'metadata': {
                    'format': 'html',
                    'content_hash': content_hash,
                    'file_size': file_path.stat().st_size
                }
            }

        except Exception as e:
            logger.error(f"Error parsing HTML {file_path}: {e}")
            raise

    @classmethod
    def parse(cls, file_path: Path) -> Optional[Dict]:
        """
        Parse file based on extension.

        Args:
            file_path: Path to document

        Returns:
            Parsed document dictionary or None if format unsupported
        """
        suffix = file_path.suffix.lower()

        parsers = {
            '.md': cls.parse_markdown,
            '.markdown': cls.parse_markdown,
            '.pdf': cls.parse_pdf,
            '.html': cls.parse_html,
            '.htm': cls.parse_html,
        }

        parser = parsers.get(suffix)
        if not parser:
            logger.warning(f"Unsupported file format: {suffix}")
            return None

        logger.info(f"Parsing {file_path} with {suffix} parser")
        return parser(file_path)


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.

    Args:
        text: Raw text

    Returns:
        Normalized text
    """
    # Collapse multiple whitespaces
    text = ' '.join(text.split())

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove excessive newlines
    while '\n\n\n' in text:
        text = text.replace('\n\n\n', '\n\n')

    return text.strip()
