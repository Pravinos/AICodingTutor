"""
Export Manager for the Interactive Coding Tutor.
Handles text and PDF export functionality.
"""

from datetime import datetime
from typing import List, Dict, Any
from io import BytesIO
import re

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Preformatted
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.colors import HexColor
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ExportManager:
    """Manages PDF export functionality for chat history"""
    
    def __init__(self):
        self.reportlab_available = REPORTLAB_AVAILABLE
    
    def export_to_pdf(self, chat_history: List[Dict[str, Any]], 
                      language: str, model: str) -> bytes:
        """
        Export chat history to PDF
        
        Args:
            chat_history: List of chat messages
            language: Programming language used
            model: AI model used
            
        Returns:
            PDF data as bytes
            
        Raises:
            ImportError: If reportlab is not available
        """
        if not self.reportlab_available:
            raise ImportError("ReportLab library is required for PDF export. Install with: pip install reportlab")
        
        buffer = BytesIO()
        # Slightly narrower margins for more content width
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=60,
            bottomMargin=40
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title', parent=styles['Heading1'], fontSize=18,
            textColor=HexColor('#2E4057'), spaceAfter=18, alignment=1
        )
        subtitle_style = ParagraphStyle(
            'Subtitle', parent=styles['Normal'], fontSize=11,
            textColor=HexColor('#555555'), spaceAfter=14, alignment=1, leading=14
        )
        meta_style = ParagraphStyle(
            'Meta', parent=styles['Normal'], fontSize=9,
            textColor=HexColor('#666666'), spaceAfter=8, alignment=1, leading=11
        )
        role_user_style = ParagraphStyle(
            'RoleUser', parent=styles['Heading4'], fontSize=11,
            textColor=HexColor('#0D47A1'), spaceBefore=10, spaceAfter=4
        )
        role_tutor_style = ParagraphStyle(
            'RoleTutor', parent=styles['Heading4'], fontSize=11,
            textColor=HexColor('#1B5E20'), spaceBefore=10, spaceAfter=4
        )
        body_style = ParagraphStyle(
            'Body', parent=styles['Normal'], fontSize=10, leading=13,
            spaceAfter=8
        )
        list_item_style = ParagraphStyle(
            'ListItem', parent=body_style, leftIndent=14, bulletIndent=4,
            spaceBefore=0, spaceAfter=2
        )
        code_block_style = ParagraphStyle(
            'CodeBlock', parent=styles['Code'] if 'Code' in styles else styles['Normal'],
            fontName='Courier', fontSize=8.5, leading=10.5,
            backColor=HexColor('#F5F5F5'), leftIndent=10, rightIndent=10,
            spaceBefore=4, spaceAfter=8, borderPadding=6
        )
        separator_style = ParagraphStyle(
            'Separator', parent=styles['Normal'], fontSize=6,
            textColor=HexColor('#999999'), alignment=1, spaceBefore=6, spaceAfter=6
        )
        
        # Build PDF content
        story = []
        
        # Title and metadata
        story.append(Paragraph("Interactive Coding Tutor", title_style))
        story.append(Paragraph("Chat History Export", subtitle_style))
        metadata = (
            f"<b>Exported:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"<b>Language:</b> {language} | <b>Model:</b> {model} | "
            f"<b>Messages:</b> {len(chat_history)}"
        )
        story.append(Paragraph(metadata, meta_style))
        story.append(Spacer(1, 10))
        
        # Chat messages
        for i, message in enumerate(chat_history):
            role = message.get("role", "unknown")
            raw = message.get("content", "")
            blocks = self._parse_markdown_like(raw)

            role_para_style = role_user_style if role == 'user' else role_tutor_style
            role_label = 'YOU' if role == 'user' else 'TUTOR'
            story.append(Paragraph(role_label, role_para_style))

            ol_index = 1
            for btype, value in blocks:
                if btype == 'p':
                    story.append(Paragraph(value, body_style))
                elif btype == 'list':
                    for item in value:
                        story.append(Paragraph(f"• {item}", list_item_style))
                elif btype == 'olist':
                    for item in value:
                        story.append(Paragraph(f"{ol_index}. {item}", list_item_style))
                        ol_index += 1
                elif btype == 'code':
                    story.append(Preformatted(value, code_block_style))
                elif btype == 'hr':
                    story.append(Paragraph("—" * 20, separator_style))

            if i < len(chat_history) - 1:
                story.append(Paragraph("", separator_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    
    # ---------------- Internal helpers ---------------- #
    def _parse_markdown_like(self, text: str) -> List[tuple]:
        """Parse simple markdown-ish content into block tuples.

        Supported blocks: paragraphs (p), unordered list (list), ordered list (olist),
        fenced code blocks (code), horizontal rule (hr).
        """
        if not text:
            return []

        lines = text.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        blocks: List[tuple] = []
        para: List[str] = []
        ulist: List[str] = []
        olist: List[str] = []
        in_code = False
        code_lines: List[str] = []

        def flush_para():
            nonlocal para
            if para:
                paragraph = ' '.join(l.strip() for l in para if l.strip())
                if paragraph:
                    blocks.append(('p', self._format_inline(paragraph)))
                para = []

        def flush_ulist():
            nonlocal ulist
            if ulist:
                blocks.append(('list', [self._format_inline(i) for i in ulist]))
                ulist = []

        def flush_olist():
            nonlocal olist
            if olist:
                blocks.append(('olist', [self._format_inline(i) for i in olist]))
                olist = []

        for line in lines:
            stripped = line.strip()

            # Code fences
            if stripped.startswith('```'):
                if in_code:
                    blocks.append(('code', '\n'.join(code_lines)))
                    code_lines = []
                    in_code = False
                else:
                    flush_para(); flush_ulist(); flush_olist()
                    in_code = True
                continue

            if in_code:
                code_lines.append(line)
                continue

            # Horizontal rule
            if re.fullmatch(r'(?:-{3,}|_{3,}|\*{3,})', stripped):
                flush_para(); flush_ulist(); flush_olist()
                blocks.append(('hr', ''))
                continue

            # Blank line -> paragraph/list boundary
            if stripped == '':
                flush_para(); flush_ulist(); flush_olist()
                continue

            # Ordered list (1. )
            if re.match(r'^\d+\.\s+', stripped):
                flush_para(); flush_ulist()
                olist.append(stripped.split('. ', 1)[1])
                continue

            # Unordered list (- * + )
            if re.match(r'^[\-*+]\s+', stripped):
                flush_para(); flush_olist()
                ulist.append(stripped[1:].strip())
                continue

            # Normal paragraph line
            if ulist:
                flush_ulist()
            if olist:
                flush_olist()
            para.append(line)

        # Final flush
        flush_para(); flush_ulist(); flush_olist()
        if in_code and code_lines:
            blocks.append(('code', '\n'.join(code_lines)))

        # Replace literal [CODE BLOCK] tokens with placeholder code block
        final: List[tuple] = []
        for btype, value in blocks:
            if btype == 'p' and '[CODE BLOCK]' in value:
                segments = [seg for seg in value.split('[CODE BLOCK]')]
                for idx, seg in enumerate(segments):
                    if seg.strip():
                        final.append(('p', seg.strip()))
                    if idx < len(segments) - 1:
                        final.append(('code', '# Code block omitted in source message'))
            else:
                final.append((btype, value))
        return final

    def _format_inline(self, text: str) -> str:
        """Apply minimal inline markdown formatting (bold, italics, code)."""
        # Escape XML first
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Bold ** **
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        # Italic * * (avoid already bold inside)
        text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
        # Inline code `code`
        text = re.sub(r'`([^`]+)`', r'<font face="Courier">\1</font>', text)
        return text
    
    def get_export_filename(self, file_type: str = 'pdf') -> str:
        """
        Generate export filename with timestamp
        
        Args:
            file_type: File type, defaults to 'pdf'
            
        Returns:
            Filename string
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"coding_tutor_chat_{timestamp}.{file_type}"
