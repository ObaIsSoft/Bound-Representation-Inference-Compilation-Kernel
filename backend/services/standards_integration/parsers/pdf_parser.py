"""
PDF Standards Parser

Parses purchased standards PDFs to extract structured data.
Supports:
- ISO standards (structured format)
- ASTM standards (test method format)
- NEC/NEMA (tabular format)
- NASA standards (technical report format)

Note: Only parses PDFs you legally own. Do not use for pirated content.
"""

from typing import Dict, Any, List, Optional, BinaryIO
from dataclasses import dataclass
from datetime import datetime
import logging
import re

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

logger = logging.getLogger(__name__)


@dataclass
class ParsedStandard:
    """Result of parsing a standards PDF"""
    standard_type: str
    standard_number: str
    revision: Optional[str]
    title: str
    publication_date: Optional[str]
    abstract: Optional[str]
    scope: Optional[str]
    tables: List[Dict[str, Any]]  # Extracted tables
    sections: Dict[str, str]  # Section name -> content
    warnings: List[str]
    

class PDFStandardParser:
    """
    Parser for standards PDFs.
    
    Extracts structured data from purchased standards documents.
    Optimized for:
    - ISO standards (tolerance tables, fit classes)
    - ASTM standards (test methods, material specs)
    - NEC (ampacity tables)
    - NASA standards (technical requirements)
    """
    
    def __init__(self):
        self.warnings = []
        
    def parse_pdf(
        self,
        pdf_path: str,
        standard_type: Optional[str] = None
    ) -> ParsedStandard:
        """
        Parse a standards PDF.
        
        Args:
            pdf_path: Path to PDF file
            standard_type: Hint for parser (iso, astm, nec, nasa)
            
        Returns:
            ParsedStandard with extracted data
        """
        self.warnings = []
        
        if not HAS_PYPDF2 and not HAS_PDFPLUMBER:
            raise ImportError(
                "PDF parsing requires PyPDF2 or pdfplumber. "
                "Run: pip install pdfplumber"
            )
        
        # Auto-detect standard type from filename if not provided
        if not standard_type:
            standard_type = self._detect_standard_type(pdf_path)
        
        # Extract text from PDF
        full_text = self._extract_text(pdf_path)
        
        # Parse based on standard type
        if standard_type.lower() in ["iso", "iec"]:
            return self._parse_iso(full_text, pdf_path)
        elif standard_type.lower() == "astm":
            return self._parse_astm(full_text, pdf_path)
        elif standard_type.lower() in ["nec", "nfpa"]:
            return self._parse_nec(full_text, pdf_path)
        elif standard_type.lower() == "nasa":
            return self._parse_nasa(full_text, pdf_path)
        else:
            return self._parse_generic(full_text, pdf_path)
    
    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        
        # Try pdfplumber first (better for tables)
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except Exception as e:
                self.warnings.append(f"pdfplumber failed: {e}")
        
        # Fall back to PyPDF2
        if HAS_PYPDF2:
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except Exception as e:
                self.warnings.append(f"PyPDF2 failed: {e}")
        
        return text
    
    def _detect_standard_type(self, pdf_path: str) -> str:
        """Auto-detect standard type from filename"""
        filename = pdf_path.lower()
        
        if "iso" in filename:
            return "iso"
        elif "astm" in filename:
            return "astm"
        elif "nec" in filename or "nfpa" in filename:
            return "nec"
        elif "nasa" in filename:
            return "nasa"
        elif "ansi" in filename:
            return "ansi"
        else:
            return "generic"
    
    def _parse_iso(self, text: str, pdf_path: str) -> ParsedStandard:
        """Parse ISO standard format"""
        # Extract standard number
        number_match = re.search(r'ISO[/\s]?IEC?\s*(\d+[-:]?\d*)', text, re.IGNORECASE)
        number = number_match.group(1) if number_match else "unknown"
        
        # Extract year/revision
        year_match = re.search(r'(\d{4})', text[:500])  # Look in first 500 chars
        revision = year_match.group(1) if year_match else None
        
        # Extract title (usually after standard number)
        title = ""
        lines = text.split('\n')[:20]  # First 20 lines
        for i, line in enumerate(lines):
            if number in line and i + 1 < len(lines):
                title = lines[i + 1].strip()
                break
        
        # Extract scope
        scope = None
        scope_match = re.search(r'(?:1\s+)?Scope\s*\n(.{100,500})', text, re.DOTALL | re.IGNORECASE)
        if scope_match:
            scope = scope_match.group(1).strip().replace('\n', ' ')
        
        # Look for tolerance tables (ISO 286)
        tables = []
        if "286" in number:  # ISO 286 is about tolerances
            tables = self._extract_iso_286_tables(text)
        
        return ParsedStandard(
            standard_type="ISO",
            standard_number=number,
            revision=revision,
            title=title,
            publication_date=revision,
            abstract=None,
            scope=scope,
            tables=tables,
            sections={},
            warnings=self.warnings
        )
    
    def _extract_iso_286_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract ISO 286 tolerance tables"""
        tables = []
        
        # Look for tolerance grade tables
        # Pattern: Grade IT6, IT7, etc. with tolerance values
        grade_pattern = r'IT(\d+)\s+([\d\.]+)\s+([\d\.]+)'
        matches = re.findall(grade_pattern, text)
        
        if matches:
            grade_data = []
            for grade, tol1, tol2 in matches[:20]:  # Limit to first 20
                grade_data.append({
                    "grade": f"IT{grade}",
                    "tolerance_um": float(tol1) if '.' in tol1 else int(tol1)
                })
            
            tables.append({
                "name": "Tolerance Grades",
                "type": "iso_tolerance_grades",
                "data": grade_data
            })
        
        # Look for fundamental deviation tables
        deviation_pattern = r'([a-zA-Z]+)\s+([\d\.]+)\s+to\s+([\d\.]+)'
        deviation_matches = re.findall(deviation_pattern, text)
        
        if deviation_matches:
            deviation_data = []
            for dev, lower, upper in deviation_matches[:20]:
                deviation_data.append({
                    "deviation": dev,
                    "range_um": f"{lower} to {upper}"
                })
            
            tables.append({
                "name": "Fundamental Deviations",
                "type": "iso_deviations",
                "data": deviation_data
            })
        
        return tables
    
    def _parse_astm(self, text: str, pdf_path: str) -> ParsedStandard:
        """Parse ASTM standard format"""
        # Extract standard number (e.g., A36/A36M-19)
        number_match = re.search(r'(A\d+[M]?(?:\/[A-Z]\d+)?)[-\s]?(\d+)?', text)
        if number_match:
            number = number_match.group(1)
            revision = number_match.group(2)
        else:
            number = "unknown"
            revision = None
        
        # Extract title
        title = ""
        lines = text.split('\n')[:30]
        for line in lines:
            if "Standard Specification" in line or "Standard Test Method" in line:
                title = line.strip()
                break
        
        # Extract scope
        scope = None
        scope_match = re.search(r'(?:1\.\s*)?Scope\s*\n(.{100,800})', text, re.DOTALL)
        if scope_match:
            scope = scope_match.group(1).strip()[:500]
        
        # Look for material property tables
        tables = self._extract_astm_tables(text)
        
        return ParsedStandard(
            standard_type="ASTM",
            standard_number=number,
            revision=revision,
            title=title,
            publication_date=revision,
            abstract=None,
            scope=scope,
            tables=tables,
            sections={},
            warnings=self.warnings
        )
    
    def _extract_astm_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract ASTM material property tables"""
        tables = []
        
        # Look for tensile strength requirements
        tensile_pattern = r'Tensile strength[^\n]*?(\d{2,6})[^\n]*?(?:MPa|ksi)'
        tensile_matches = re.findall(tensile_pattern, text, re.IGNORECASE)
        
        if tensile_matches:
            tables.append({
                "name": "Tensile Strength Requirements",
                "type": "mechanical_properties",
                "data": [{"tensile_strength": m} for m in tensile_matches[:10]]
            })
        
        # Look for chemical composition tables
        chem_pattern = r'(C|Mn|P|S|Si)\s*≤?\s*(\d+\.?\d*)\s*%'
        chem_matches = re.findall(chem_pattern, text)
        
        if chem_matches:
            composition = {elem: f"{val}%" for elem, val in chem_matches[:10]}
            tables.append({
                "name": "Chemical Composition",
                "type": "chemical_composition",
                "data": [composition]
            })
        
        return tables
    
    def _parse_nec(self, text: str, pdf_path: str) -> ParsedStandard:
        """Parse NEC (National Electrical Code) format"""
        # NEC tables are usually ampacity tables
        tables = []
        
        # Look for Table 310.16 (common ampacity table)
        if "310.16" in text:
            ampacity_data = self._extract_nec_ampacity_table(text)
            if ampacity_data:
                tables.append({
                    "name": "NEC Table 310.16",
                    "type": "ampacity",
                    "data": ampacity_data
                })
        
        return ParsedStandard(
            standard_type="NEC",
            standard_number="NFPA 70",
            revision=None,
            title="National Electrical Code",
            publication_date=None,
            abstract="National Electrical Code ampacity tables",
            scope="Electrical wiring and equipment",
            tables=tables,
            sections={},
            warnings=self.warnings
        )
    
    def _extract_nec_ampacity_table(self, text: str) -> List[Dict[str, Any]]:
        """Extract NEC ampacity table data"""
        data = []
        
        # Pattern: AWG size followed by ampacities
        # Example: "14 15 20 25" -> AWG 14: 15A@60C, 20A@75C, 25A@90C
        ampacity_pattern = r'(\d{1,2})/0?\s+(\d{1,3})\s+(\d{1,3})\s+(\d{1,3})'
        matches = re.findall(ampacity_pattern, text)
        
        for awg, amp60, amp75, amp90 in matches[:20]:
            data.append({
                "awg": awg,
                "ampacity_60c_a": int(amp60),
                "ampacity_75c_a": int(amp75),
                "ampacity_90c_a": int(amp90)
            })
        
        return data
    
    def _parse_nasa(self, text: str, pdf_path: str) -> ParsedStandard:
        """Parse NASA standard format"""
        # Extract NASA-STD number
        number_match = re.search(r'NASA-STD-(\d{4}[A-Z]?)', text)
        number = number_match.group(1) if number_match else "unknown"
        
        # Extract revision from number
        revision = None
        if number and number[-1].isalpha():
            revision = number[-1]
            number = number[:-1]
        
        # Extract title
        title = ""
        lines = text.split('\n')[:30]
        for line in lines:
            if "Standard" in line and len(line) > 20:
                title = line.strip()
                break
        
        return ParsedStandard(
            standard_type="NASA",
            standard_number=number,
            revision=revision,
            title=title,
            publication_date=None,
            abstract=None,
            scope=None,
            tables=[],
            sections={},
            warnings=self.warnings
        )
    
    def _parse_generic(self, text: str, pdf_path: str) -> ParsedStandard:
        """Generic parser for unknown standard types"""
        return ParsedStandard(
            standard_type="generic",
            standard_number="unknown",
            revision=None,
            title="",
            publication_date=None,
            abstract=None,
            scope=None,
            tables=[],
            sections={},
            warnings=self.warnings + ["Could not detect standard type"]
        )
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all tables from a PDF using pdfplumber.
        
        Returns:
            List of tables as list of dicts
        """
        if not HAS_PDFPLUMBER:
            raise ImportError("pdfplumber required for table extraction")
        
        tables = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                
                for table in page_tables:
                    if table and len(table) > 1:  # Has header + data
                        # Convert to list of dicts
                        headers = table[0]
                        rows = []
                        
                        for row in table[1:]:
                            row_dict = {}
                            for j, cell in enumerate(row):
                                if j < len(headers):
                                    row_dict[headers[j]] = cell
                            rows.append(row_dict)
                        
                        tables.append({
                            "page": i + 1,
                            "headers": headers,
                            "data": rows
                        })
        
        return tables
