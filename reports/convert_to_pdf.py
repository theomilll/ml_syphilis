#!/usr/bin/env python3
"""
Script to convert the Markdown report to PDF with proper formatting.
Requires: pip install markdown2 weasyprint
"""

import markdown2
import os
from weasyprint import HTML, CSS

def convert_md_to_pdf(md_file, pdf_file):
    """Convert a markdown file to PDF with styling."""
    
    # Read the markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown2.markdown(
        md_content,
        extras=['tables', 'fenced-code-blocks', 'header-ids']
    )
    
    # Add HTML structure and CSS styling
    full_html = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <title>Relatório Final - Predição de Sífilis Congênita</title>
        <style>
            @page {{
                size: A4;
                margin: 2.5cm;
            }}
            
            body {{
                font-family: 'Times New Roman', serif;
                font-size: 12pt;
                line-height: 1.6;
                color: #333;
                text-align: justify;
            }}
            
            h1 {{
                font-size: 24pt;
                margin-top: 0;
                margin-bottom: 20pt;
                text-align: center;
                color: #1a1a1a;
            }}
            
            h2 {{
                font-size: 18pt;
                margin-top: 24pt;
                margin-bottom: 12pt;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5pt;
            }}
            
            h3 {{
                font-size: 14pt;
                margin-top: 18pt;
                margin-bottom: 9pt;
                color: #34495e;
            }}
            
            h4 {{
                font-size: 12pt;
                margin-top: 12pt;
                margin-bottom: 6pt;
                color: #34495e;
                font-weight: bold;
            }}
            
            p {{
                margin-bottom: 12pt;
                text-indent: 1.5em;
            }}
            
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 15pt 0;
                font-size: 10pt;
            }}
            
            th, td {{
                border: 1px solid #ddd;
                padding: 8pt;
                text-align: left;
            }}
            
            th {{
                background-color: #3498db;
                color: white;
                font-weight: bold;
            }}
            
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            
            ul, ol {{
                margin-bottom: 12pt;
                padding-left: 30pt;
            }}
            
            li {{
                margin-bottom: 6pt;
            }}
            
            code {{
                background-color: #f4f4f4;
                padding: 2pt 4pt;
                font-family: 'Courier New', monospace;
                font-size: 10pt;
            }}
            
            blockquote {{
                margin: 15pt 0;
                padding: 10pt 20pt;
                background-color: #f9f9f9;
                border-left: 4px solid #3498db;
            }}
            
            hr {{
                margin: 24pt 0;
                border: none;
                border-top: 2px solid #e0e0e0;
            }}
            
            strong {{
                font-weight: bold;
                color: #2c3e50;
            }}
            
            /* Page numbers */
            @page {{
                @bottom-right {{
                    content: counter(page);
                    font-size: 10pt;
                }}
            }}
            
            /* Keep headings with their content */
            h1, h2, h3, h4 {{
                page-break-after: avoid;
            }}
            
            /* Avoid breaking inside tables */
            table {{
                page-break-inside: avoid;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    HTML(string=full_html).write_pdf(pdf_file)
    print(f"PDF generated successfully: {pdf_file}")

if __name__ == "__main__":
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    md_file = os.path.join(script_dir, "final_report.md")
    pdf_file = os.path.join(script_dir, "final_report.pdf")
    
    # Convert
    convert_md_to_pdf(md_file, pdf_file)