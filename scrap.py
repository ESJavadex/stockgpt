from flask import Flask, render_template, request, send_file, make_response, session, g
import requests
from bs4 import BeautifulSoup
import os
from openai import OpenAI
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import io
import matplotlib
matplotlib.use('Agg')  # ensure no GUI backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
from datetime import datetime
import pdfkit
from textblob import TextBlob
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from yahooquery import Ticker
import json
from requests_html import HTMLSession
import base64
from concurrent.futures import ThreadPoolExecutor
import time
import tempfile
from curl_cffi import requests as curl_requests
from flask_session import Session  # Add Flask-Session
import hashlib

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure session handling
app.config.update(
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(days=1),  # Sessions last 1 day
    SESSION_TYPE="filesystem",  # Use filesystem instead of signed cookies
    SESSION_FILE_DIR=os.path.join(os.getcwd(), "flask_session")  # Store sessions in the flask_session directory
)

# Generate a secure secret key
if not os.environ.get("SECRET_KEY"):
    # Generate a random key for development
    import secrets
    os.environ["SECRET_KEY"] = secrets.token_hex(16)

app.secret_key = os.environ.get("SECRET_KEY")

# Initialize Flask-Session
Session(app)

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

import markdown
import re

def chat_with_gpt(prompt):
    print("[DEBUG] OpenAI API Key Loaded:", bool(os.environ.get("OPENAI_API_KEY")))
    # Increased max prompt length since GPT-4.1 supports 100k tokens
    max_prompt_length = 50000  # Significantly increased from 12000
    if len(prompt) > max_prompt_length:
        print(f"[DEBUG] Prompt too long ({len(prompt)} chars), truncating to {max_prompt_length} chars")
        # Keep the beginning and end of the prompt
        start_portion = prompt[:25000]  # Keep the first 25000 chars
        end_portion = prompt[-25000:]   # Keep the last 25000 chars
        prompt = start_portion + "\n\n[...some content truncated for brevity...]\n\n" + end_portion
    
    print("[DEBUG] Sending prompt to GPT-4.1 (length: " + str(len(prompt)) + " chars)")
    
    try:
        # Set a timeout for the API call
        chat_completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=20000,  # Increased max tokens for response
            timeout=180  # 3 minute timeout
        )
        print("[DEBUG] Received response from GPT")
        
        if hasattr(chat_completion, 'choices') and chat_completion.choices:
            gpt_response = chat_completion.choices[0].message.content
            print("[DEBUG] GPT Response length:", len(gpt_response))
        else:
            print("[ERROR] No choices in chat_completion:", chat_completion)
            gpt_response = "[No response from GPT-4.1]"
    except Exception as e:
        print("[ERROR] Exception during OpenAI GPT call:", e)
        import traceback
        traceback.print_exc()
        # Provide a more user-friendly error message
        return f"""
        <div class="alert alert-danger">
            <h4>Error communicating with AI service</h4>
            <p>There was a problem generating the analysis. This could be due to:</p>
            <ul>
                <li>Service temporarily unavailable</li>
                <li>Request timeout (the analysis takes too long)</li>
                <li>API limit reached</li>
            </ul>
            <p>Error details: {str(e)}</p>
            <p>Please try again in a few minutes with a different stock or contact support if the problem persists.</p>
        </div>
        """
    
    # Process the response to improve formatting
    processed_response = preprocess_markdown(gpt_response)
    print("[DEBUG] Processed Response length:", len(processed_response))
    
    # Special handling for stock analysis format: Convert patterns like # SECTION TITLE to proper headers
    # This handles patterns like # 1. VALUATION SUMMARY or #1. VALUATION SUMMARY
    processed_response = re.sub(r'(?m)^# *(\d+\. .+)$', r'## \1', processed_response)
    processed_response = re.sub(r'(?m)^#(\d+\. .+)$', r'## \1', processed_response)
    # Add proper spacing to section headers (e.g., #SUMMARY to # SUMMARY)
    processed_response = re.sub(r'(?m)^#([A-Z])', r'# \1', processed_response)
    
    # Handle stock summary formats often seen in the responses
    processed_response = re.sub(r'\*\*([^*]+):\*\*', r'**\1:** ', processed_response)
    
    try:
        # Import pymdown extensions
        import pymdownx.superfences
        import pymdownx.highlight
        import pymdownx.inlinehilite
        import pymdownx.keys
        import pymdownx.smartsymbols
        import pymdownx.tabbed
        
        # Try to import math extension
        try:
            import mdx_math
            has_math_extension = True
        except ImportError:
            has_math_extension = False
            print("[WARNING] mdx_math extension not available, math expressions won't be rendered")
        
        # Prepare extensions list
        extensions = [
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.codehilite',  # Syntax highlighting for code blocks
            'markdown.extensions.nl2br',       # Convert newlines to <br>
            'markdown.extensions.sane_lists',  # Better list handling
            'markdown.extensions.attr_list',   # Allow attributes in markdown
            'markdown.extensions.def_list',    # Definition lists support
            'markdown.extensions.smarty',      # Smart quotes, dashes, etc.
            'markdown.extensions.meta',        # Metadata extraction
            'pymdownx.superfences',            # Nested code blocks
            'pymdownx.highlight',              # Better code highlighting
            'pymdownx.inlinehilite',           # Inline code highlighting
            'pymdownx.keys',                   # Keyboard key display
            'pymdownx.smartsymbols',           # Smart symbols (fractions, arrows)
            'pymdownx.emoji',                  # Emoji support
            'pymdownx.tasklist',               # Task list support
            'pymdownx.mark',                   # Highlighted text
            'pymdownx.tilde'                   # Strike-through
        ]
        
        # Add math extension if available
        if has_math_extension:
            extensions.append('mdx_math')  # Math expressions support
        
        # Prepare extension configs
        extension_configs = {
            'pymdownx.highlight': {
                'use_pygments': True,
                'linenums': False
            },
            'pymdownx.superfences': {
                'custom_fences': [
                    {'name': 'flow', 'class': 'uml-flowchart', 'format': pymdownx.superfences.fence_code_format},
                    {'name': 'sequence', 'class': 'uml-sequence-diagram', 'format': pymdownx.superfences.fence_code_format}
                ]
            }
        }
        
        # Add math config if available
        if has_math_extension:
            extension_configs['mdx_math'] = {
                'enable_dollar_delimiter': True,  # Enable $...$ as inline math delimiter
                'add_preview': True               # Add preview class for better styling
            }
        
        # Use a more robust set of extensions for markdown conversion
        html_response = markdown.markdown(
            processed_response,
            extensions=extensions,
            extension_configs=extension_configs
        )
        print("[DEBUG] Markdown to HTML conversion successful")
        
        # Further enhance the HTML with additional styling
        html_response = postprocess_html(html_response)
        return html_response
    except Exception as e:
        print("[ERROR] Error in pymdown markdown processing:", e)
        import traceback
        traceback.print_exc()
        
        # Try a more basic conversion as fallback
        try:
            # Simpler conversion with just essential extensions
            html_response = markdown.markdown(
                processed_response,
                extensions=[
                    'markdown.extensions.tables',
                    'markdown.extensions.fenced_code',
                    'markdown.extensions.nl2br'
                ]
            )
            html_response = postprocess_html(html_response)
            return html_response
        except Exception as e2:
            print("[ERROR] Fallback markdown processing also failed:", e2)
            # Return a simpler version if markdown processing fails
            return f"""
            <div class="alert alert-warning">
                <h4>Formatting Error</h4>
                <p>The analysis was generated but could not be properly formatted.</p>
                <p>Error details: {str(e)}</p>
                <pre style="white-space: pre-wrap;">{gpt_response}</pre>
            </div>
            """

def preprocess_markdown(text):
    """Preprocess markdown to improve formatting before conversion"""
    # First check if entire response is wrapped in a markdown code block
    # This handles cases where the LLM outputs ```markdown or ```md
    markdown_code_block_pattern = re.compile(r'^```(?:markdown|md)?\s*\n(.*?)\n```\s*$', re.DOTALL)
    match = markdown_code_block_pattern.match(text.strip())
    if match:
        # Extract the content from within the code block
        text = match.group(1)
        print("[DEBUG] Extracted markdown from code block")
    
    # Regular code block handling
    code_blocks = {}
    code_block_pattern = re.compile(r'```(.*?)\n(.*?)```', re.DOTALL)
    
    def save_code_block(match):
        language = match.group(1).strip()
        code = match.group(2)
        
        # If the language is markdown/md, we should process this content as markdown, not code
        if language.lower() in ['markdown', 'md']:
            return code  # Return content directly, don't save as code block
            
        placeholder = f"CODE_BLOCK_{len(code_blocks)}"
        code_blocks[placeholder] = (language, code)
        return placeholder
    
    # Save code blocks and replace with placeholders
    text = code_block_pattern.sub(save_code_block, text)
    
    # Handle missing space after # for headings - ensure proper Markdown heading format
    lines = text.split('\n')
    for i in range(len(lines)):
        # Match patterns like "#Heading" and convert to "# Heading"
        if re.match(r'^#+[^#\s]', lines[i]):
            lines[i] = re.sub(r'^(#+)([^#\s])', r'\1 \2', lines[i])
        
        # Ensure blank line before headings for proper rendering
        if i > 0 and re.match(r'^#+\s', lines[i]) and lines[i-1].strip() != '':
            lines[i] = '\n' + lines[i]
    
    text = '\n'.join(lines)
    
    # Fix common markdown formatting errors
    
    # Fix invalid emphasis/strong markers
    text = re.sub(r'(\w)\*\*(\w)', r'\1 **\2', text)  # Add space before **
    text = re.sub(r'(\w)\*(\w)', r'\1 *\2', text)     # Add space before *
    text = re.sub(r'(\*\*\w+)\*\*(\w)', r'\1** \2', text)  # Add space after **
    text = re.sub(r'(\*\w+)\*(\w)', r'\1* \2', text)      # Add space after *
    
    # Fix inline code markers
    text = re.sub(r'(\w)`(\w)', r'\1 `\2', text)      # Add space before `
    text = re.sub(r'(`\w+)`(\w)', r'\1` \2', text)    # Add space after `
    
    # Fix links with missing spaces
    text = re.sub(r'(\w)\[(.*?)\]', r'\1 [\2]', text)
    
    # Fix CSV-like output by converting it to proper markdown tables
    lines = text.split('\n')
    processed_lines = []
    in_csv_section = False
    csv_lines = []
    table_row_pattern = re.compile(r'^\s*\|.*\|\s*$')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Detect CSV section start - specific for trading plan section
        if 'stock_name,resistance_levels,support_levels,entry_price,target_sell_range,stop_loss_levels' in line:
            in_csv_section = True
            csv_lines = [line]
            i += 1
            continue
        
        # If we're in a CSV section and encounter an empty line or non-CSV line, convert and end the section
        if in_csv_section and (not line.strip() or not ',' in line):
            in_csv_section = False
            if len(csv_lines) > 0:
                # Convert CSV to markdown table
                md_table = convert_csv_to_md_table(csv_lines)
                processed_lines.append(md_table)
                processed_lines.append('')  # Add empty line after table
            if line.strip():  # If this line wasn't empty, add it
                processed_lines.append(line)
            i += 1
            continue
        
        # Add line to CSV collection if in CSV section
        if in_csv_section:
            csv_lines.append(line)
            i += 1
            continue
        
        # Check for potential table data but not properly formatted as markdown table
        # This handles misaligned or improperly formatted tables
        if '|' in line and not table_row_pattern.match(line) and not line.strip().startswith('#'):
            # If it has pipe characters but not formatted properly as a markdown table row
            # Check if we have multiple columns to form a table
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:  # At least 3 parts (including empty edges) indicates a table row
                # Format as a proper markdown table row
                processed_lines.append('| ' + ' | '.join([cell.strip() for cell in parts if cell.strip()]) + ' |')
                i += 1
                continue
        
        # Special handling for executive summary section - look for patterns like "Entry price: 123.45"
        if re.match(r'^\s*(Entry price|Trade direction|Take Profit|Stop Loss|Risk/Reward|Time Horizon|Conviction Level)\s*:\s*', line, re.IGNORECASE):
            # Collect all the executive summary items and format as a table
            summary_items = []
            while i < len(lines) and (i == 0 or lines[i].strip() or i == len(lines)-1):
                current_line = lines[i].strip()
                if current_line and re.match(r'^\s*(Entry price|Trade direction|Take Profit|Stop Loss|Risk/Reward|Time Horizon|Conviction Level)\s*:\s*', current_line, re.IGNORECASE):
                    # Found an executive summary item
                    summary_items.append(current_line)
                elif summary_items and not current_line:  # End of section
                    break
                i += 1
                
            if summary_items:
                # Create a markdown table from the summary items
                processed_lines.append('')  # Add empty line before table
                processed_lines.append('### Executive Summary Table')
                processed_lines.append('')
                processed_lines.append('| Parameter | Value |')
                processed_lines.append('|---|---|')
                for item in summary_items:
                    match = re.match(r'^\s*(.*?)\s*:\s*(.*)$', item)
                    if match:
                        key, value = match.groups()
                        processed_lines.append(f'| {key.strip()} | {value.strip()} |')
                processed_lines.append('')  # Add empty line after table
                continue
        
        # Fix table rows with <tr><td> in the text (happens when LLM outputs HTML instead of markdown)
        # This pattern detects text that looks like <tr><td>content</td><td>more content</td></tr>
        if '<tr><td>' in line and '</td></tr>' in line:
            # Convert directly to proper markdown table row format
            line = line.replace('<tr><td>', '| ')
            line = line.replace('</td><td>', ' | ')
            line = line.replace('</td></tr>', ' |')
            processed_lines.append(line)
            i += 1
            continue
        
        # Fix summary table formatting
        if 'Entry Price' in line and 'Direction' in line and 'TP' in line and 'SL' in line:
            processed_lines.append('')  # Add empty line before table
            processed_lines.append('### Summary Table')
            processed_lines.append('')
            # Start a proper markdown table
            headers = [h.strip() for h in line.split('|') if h.strip()]
            if headers:  # Ensure we have actual headers
                processed_lines.append('| ' + ' | '.join(headers) + ' |')
                processed_lines.append('|' + '---|' * len(headers))
            i += 1
            continue
        
        # Skip separator rows that are just dashes
        if line.count('-') > 10 and '|' in line and line.replace('-', '').replace('|', '').strip() == '':
            i += 1
            continue
        
        # Detect potential table rows with multiple pipe symbols
        if line.count('|') >= 2:  # At least one column (2 pipe symbols)
            # Clean and standardize the table row format
            cells = [cell.strip() for cell in line.split('|')]
            # Remove empty cells at the start/end
            if cells and not cells[0].strip():
                cells = cells[1:]
            if cells and not cells[-1].strip():
                cells = cells[:-1]
                
            if cells:  # If we have actual cells
                processed_lines.append('| ' + ' | '.join(cells) + ' |')
            i += 1
            continue
        
        # Normal line, just add it
        processed_lines.append(line)
        i += 1
    
    # If we ended the file still in CSV mode, convert it
    if in_csv_section and csv_lines:
        md_table = convert_csv_to_md_table(csv_lines)
        processed_lines.append(md_table)
    
    # Post-process the joined text
    text = '\n'.join(processed_lines)
    
    # Fix common markdown formatting issues
    # Ensure proper spacing for lists
    text = re.sub(r'(\n[0-9]+\.)([^\s])', r'\1 \2', text)  # Add space after numbered list markers
    text = re.sub(r'(\n- )([^\s])', r'\1\2', text)  # Ensure proper formatting for bullet lists
    
    # Fix list continuation - ensure proper indentation
    lines = text.split('\n')
    for i in range(1, len(lines)):
        # If the previous line is a list item and this line is not a list item, preserve list structure
        if (re.match(r'^\s*[0-9]+\.', lines[i-1]) or re.match(r'^\s*- ', lines[i-1])) and not re.match(r'^\s*$', lines[i]) and not re.match(r'^\s*[0-9]+\.', lines[i]) and not re.match(r'^\s*- ', lines[i]) and not re.match(r'^\s*#+', lines[i]):
            # It's a continuation of a list item, indent it properly
            if not lines[i].startswith('    '):
                lines[i] = '    ' + lines[i]
    
    text = '\n'.join(lines)
    
    # Ensure tables have proper formatting - check for tables without proper markdown formatting
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        # If line contains multiple pipe characters but isn't properly formatted
        if '|' in lines[i] and lines[i].count('|') >= 3 and not lines[i].strip().startswith('|'):
            lines[i] = '| ' + lines[i].replace('|', ' | ').strip() + ' |'
        # Look for consecutive lines with pipe characters but missing header separator
        if i > 0 and '|' in lines[i-1] and '|' in lines[i] and lines[i-1].startswith('|') and lines[i].startswith('|'):
            # Check if this looks like a header row followed by content row without separator
            if not any(c == '-' for c in lines[i]) and i < len(lines)-1 and '|' in lines[i+1]:
                # Count cells to determine how many separator cells to insert
                cell_count = lines[i-1].count('|') - 1
                # Insert separator row
                lines.insert(i, '|' + '---|' * cell_count)
                # Skip the newly inserted line in the next iteration
                i += 1
        i += 1
    
    text = '\n'.join(lines)
    
    # Add blank lines around blockquotes for proper rendering
    text = re.sub(r'([^\n])\n>(.*?)$', r'\1\n\n>\2', text, flags=re.MULTILINE)
    
    # Add blank lines around lists for proper rendering
    text = re.sub(r'([^\n])\n(- .*?)$', r'\1\n\n\2', text, flags=re.MULTILINE)
    text = re.sub(r'([^\n])\n([0-9]+\. .*?)$', r'\1\n\n\2', text, flags=re.MULTILINE)
    
    # Restore code blocks
    for placeholder, (language, code) in code_blocks.items():
        if language:
            text = text.replace(placeholder, f"```{language}\n{code}```")
        else:
            text = text.replace(placeholder, f"```\n{code}```")
    
    # Convert any remaining # headers without proper markdown format
    # For example: "#1. VALUATION SUMMARY" should be "## 1. VALUATION SUMMARY"
    text = re.sub(r'(?m)^#(\d+\.)', r'## \1', text)
    
    # Ensure proper spacing for section headings
    # Convert patterns like "#SECTION TITLE" to "# SECTION TITLE"
    text = re.sub(r'(?m)^(#+)([A-Z])', r'\1 \2', text)
    
    # Fix for dollars and percentages (ensure space after numbers)
    text = re.sub(r'(\$[0-9,.]+)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([0-9]+%)([A-Za-z])', r'\1 \2', text)
    
    return text

def convert_csv_to_md_table(csv_lines):
    """Convert CSV lines to a markdown table"""
    if not csv_lines:
        return ''

    # Extract headers and create markdown table header
    headers = [h.strip() for h in csv_lines[0].split(',')]
    md_table = '| ' + ' | '.join(headers) + ' |\n'
    md_table += '|' + '---|' * len(headers) + '\n'

    # Add table rows
    for row in csv_lines[1:]:
        if not row.strip():
            continue
        cells = [c.strip() for c in row.split(',')]
        # Ensure the number of cells matches headers
        if len(cells) < len(headers):
            cells += [''] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        md_table += '| ' + ' | '.join(cells) + ' |\n'

    return md_table

def postprocess_html(html):
    """Enhance HTML with additional styling and fixes"""
    # First fix table generation issues - we want proper <table> not just text
    # Find table rows that might not be properly formatted in HTML (often from markdown conversion issues)
    html = re.sub(r'\|\s*([^\|]+)\s*\|\s*([^\|]+)\s*\|', r'<tr><td>\1</td><td>\2</td></tr>', html)
    
    # Check if we still have HTML tags as literal text (like "<tr><td>") in the output
    # This happens when the LLM outputs HTML tags directly instead of markdown
    if re.search(r'&lt;tr&gt;&lt;td&gt;', html) or re.search(r'<tr><td>', html):
        # Extract any HTML table fragments and convert them to proper tables
        table_fragment_pattern = re.compile(r'(?:<tr><td>|&lt;tr&gt;&lt;td&gt;)(.*?)(?:</td></tr>|&lt;/td&gt;&lt;/tr&gt;)', re.DOTALL)
        
        def convert_table_fragment(match):
            content = match.group(1)
            # Handle escaped HTML
            if '&lt;' in content:
                content = content.replace('&lt;/td&gt;&lt;td&gt;', '</td><td>')
                content = content.replace('&lt;td&gt;', '<td>')
                content = content.replace('&lt;/td&gt;', '</td>')
            else:
                content = content.replace('</td><td>', '</td><td>')
            
            return f"<tr><td>{content}</td></tr>"
        
        html = table_fragment_pattern.sub(convert_table_fragment, html)
    
    # If we have a highlighted code block that contains the entire stock analysis,
    # we need to remove it and parse its contents properly
    if '<div class="highlight"><pre><span></span><code>' in html:
        # Extract the content from the highlighted code block
        pattern = re.compile(r'<div class="highlight"><pre><span></span><code>(.*?)</code></pre></div>', re.DOTALL)
        match = pattern.search(html)
        
        if match:
            # Get the content of the code block
            code_content = match.group(1)
            
            # Check if it looks like it should be rendered as markdown instead of code
            if (code_content.count('<span class="gh">') > 0 or 
                code_content.count('<span class="gu">') > 0 or
                '<tr><td>' in code_content):
                
                # This is likely markdown content that was put in a code block
                # First, convert the syntax highlighting spans to their HTML equivalents
                code_content = code_content.replace('<span class="gh">', '<h1>')
                code_content = code_content.replace('</span>', '</h1>')
                code_content = code_content.replace('<span class="gu">', '<h2>')
                code_content = code_content.replace('</span>', '</h2>')
                code_content = code_content.replace('<span class="k">', '<strong>')
                code_content = code_content.replace('</span>', '</strong>')
                code_content = code_content.replace('<span class="gs">', '<strong>')
                code_content = code_content.replace('</span>', '</strong>')
                
                # Convert literal table markup to HTML tables
                code_content = re.sub(r'<tr><td>(.*?)</td><td>(.*?)</td></tr>', 
                                    r'<tr><td>\1</td><td>\2</td></tr>', code_content)
                
                # Fix any broken tables - ensure they're wrapped in table tags
                if '<tr>' in code_content and '<table>' not in code_content:
                    code_content = re.sub(r'(<tr>.*?</tr>)', r'<table class="table table-striped">\1</table>', 
                                         code_content, flags=re.DOTALL)
                
                # Replace the original highlighted code block with our processed content
                html = html.replace(match.group(0), code_content)
    
    # Clean up any existing table classes first
    html = re.sub(r'<table class="[^"]*"', '<table', html)
    
    # Add Bootstrap table classes
    html = html.replace('<table>', '<table class="table table-striped table-bordered">')
    
    # Add special classes to different types of tables
    # Trading plan table (section 5)
    if 'TRADING PLAN' in html:
        html = re.sub(r'(<h[1-6][^>]*>\s*(?:5\.|TRADING PLAN)[^<]*</h[1-6]>[\s\S]*?)<table', 
                    r'\1<table class="trading-plan table table-striped table-bordered"', html, flags=re.IGNORECASE)
    
    # Summary table
    if 'Summary Table' in html:
        html = re.sub(r'(<h[1-6][^>]*>\s*Summary Table[^<]*</h[1-6]>[\s\S]*?)<table', 
                    r'\1<table class="summary-table table table-striped table-bordered"', html, flags=re.IGNORECASE)
    
    # Executive summary table
    if 'EXECUTIVE SUMMARY' in html:
        html = re.sub(r'(<h[1-6][^>]*>\s*(?:6\.|EXECUTIVE SUMMARY)[^<]*</h[1-6]>[\s\S]*?)<table', 
                    r'\1<table class="summary-table table table-striped table-bordered"', html, flags=re.IGNORECASE)
    
    # Fix duplicate class attributes
    html = re.sub(r'class="([^"]*)"\s+class="([^"]*)"', r'class="\1 \2"', html)
    
    # Ensure table structure is complete
    # Check if there are <tr> tags without proper <tbody> or <thead>
    if '<tr>' in html and '<tbody>' not in html:
        html = html.replace('<table', '<table')
        html = re.sub(r'(<table[^>]*>)\s*(<tr>)', r'\1<tbody>\2', html)
        html = re.sub(r'(</tr>)\s*(</table>)', r'\1</tbody>\2', html)

    # Fix code blocks - ensure they have proper styling and structure
    html = re.sub(r'<pre><code>(.*?)</code></pre>', r'<pre class="code-block"><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # Add syntax highlighting classes to code blocks with language specification
    # This regex finds language specifications like ```python or ```javascript
    code_lang_pattern = re.compile(r'<pre><code class="language-(\w+)">(.*?)</code></pre>', re.DOTALL)
    html = code_lang_pattern.sub(r'<pre class="language-\1"><code class="language-\1">\2</code></pre>', html)
    
    # Preserve newlines within pre and code blocks
    chunks = []
    current_position = 0
    for match in re.finditer(r'(<pre>.*?</pre>|<code>.*?</code>)', html, re.DOTALL):
        # Add the text before the match with replaced newlines
        chunks.append(html[current_position:match.start()].replace('\n', ' '))
        # Add the pre/code block as-is, preserving its newlines
        chunks.append(match.group(0))
        current_position = match.end()
    # Add any remaining text after the last match
    if current_position < len(html):
        chunks.append(html[current_position:].replace('\n', ' '))
    html = ''.join(chunks)
    
    # Fix any potentially broken links
    html = re.sub(r'<a\s+href="(.*?)"\s*>(.*?)</a>', r'<a href="\1" target="_blank" rel="noopener noreferrer">\2</a>', html)
    
    # Add styling to blockquotes
    html = html.replace('<blockquote>', '<blockquote class="blockquote border-left pl-3">')
    
    # Remove excessive <br> tags
    html = re.sub(r'<br\s*/?><br\s*/?>',  '<br>', html)
    html = re.sub(r'(<br\s*/?>\s*){2,}', '<br>', html)
    
    # Fix spacing in lists
    html = re.sub(r'<li><br>', r'<li>', html)
    html = re.sub(r'<br></li>', r'</li>', html)
    
    # Fix spacing in tables
    html = re.sub(r'<table\s+class="[^"]*"><br>', r'<table class="table table-striped table-bordered">', html)
    html = re.sub(r'<thead><br>', r'<thead>', html)
    html = re.sub(r'<tbody><br>', r'<tbody>', html)
    html = re.sub(r'<tr><br>', r'<tr>', html)
    html = re.sub(r'<td><br>', r'<td>', html)
    html = re.sub(r'<th><br>', r'<th>', html)
    html = re.sub(r'<br></tr>', r'</tr>', html)
    html = re.sub(r'<br></td>', r'</td>', html)
    html = re.sub(r'<br></th>', r'</th>', html)
    html = re.sub(r'<br></tbody>', r'</tbody>', html)
    html = re.sub(r'<br></thead>', r'</thead>', html)
    html = re.sub(r'<br></table>', r'</table>', html)
    
    # Fix spacing in headings
    html = re.sub(r'<h([1-6])><br>', r'<h\1>', html)
    html = re.sub(r'<br></h([1-6])>', r'</h\1>', html)
    
    # Fix spacing in paragraphs
    html = re.sub(r'<p><br>', r'<p>', html)
    html = re.sub(r'<br></p>', r'</p>', html)
    
    # Fix spacing in code blocks
    html = re.sub(r'<pre><br>', r'<pre>', html)
    html = re.sub(r'<code><br>', r'<code>', html)
    html = re.sub(r'<br></code>', r'</code>', html)
    html = re.sub(r'<br></pre>', r'</pre>', html)
    
    # Ensure table structure is valid
    # Missing closing tags
    html = re.sub(r'<tr>([^<]*)</tr>(?!\s*<tr>|\s*</tbody>|\s*</table>)', r'<tr><td>\1</td></tr>', html)
    html = re.sub(r'<tr>((?:(?!</tr>).)*?)$', r'<tr>\1</tr>', html) # Fix unclosed tr tags
    html = re.sub(r'<table[^>]*>\s*((?:(?!</table>).)*?)$', r'<table class="table">\1</table>', html) # Fix unclosed table tags
    
    # Remove duplicate headings that are next to each other
    html = re.sub(r'<h([1-6])\s+class="[^"]*">(.*?)</h\1>\s*<h\1\s+class="[^"]*">\2</h\1>', 
                 r'<h\1 class="text-info">\2</h\1>', html)
    
    # Handle syntax highlighting classes
    # Make sure span.gh elements are treated as headers
    html = re.sub(r'<span class="gh">(.*?)</span>', 
                 r'<h1 class="text-primary font-weight-bold">\1</h1>', html)
    
    # Make sure span.gu elements are treated as subheaders
    html = re.sub(r'<span class="gu">(.*?)</span>', 
                 r'<h2 class="text-secondary font-weight-bold">\1</h2>', html)
    
    # Convert span.k (bullet points) to proper list items
    html = re.sub(r'<span class="k">-</span><span class="w">\s*</span>(.*?)(?=<span class="k">-</span>|<span class="gu">|$)', 
                 r'<ul class="list-group list-group-flush"><li class="list-group-item">\1</li></ul>', html, flags=re.DOTALL)
    
    # Fix span.gs (strong text) elements
    html = re.sub(r'<span class="gs">(.*?)</span>', 
                 r'<strong class="text-success">\1</strong>', html)
    
    # Enhance headings with Bootstrap classes
    html = re.sub(r'<h1>(.*?)</h1>', r'<h1 class="text-primary font-weight-bold">\1</h1>', html)
    html = re.sub(r'<h2>(.*?)</h2>', r'<h2 class="text-secondary font-weight-bold">\1</h2>', html)
    html = re.sub(r'<h3>(.*?)</h3>', r'<h3 class="text-info font-weight-bold">\1</h3>', html)
    
    # Style the Executive Summary section
    html = re.sub(r'<h[1-6]\s+class="[^"]*">\s*Executive Summary\s*</h[1-6]>', 
                 r'<div class="card mb-4"><div class="card-header bg-primary text-white font-weight-bold">Executive Summary</div><div class="card-body">', 
                 html, flags=re.IGNORECASE)
    if 'Executive Summary' in html:
        # Find the next heading to close the card
        next_heading = re.search(r'<h[1-6]', html[html.find('Executive Summary') + 20:])
        if next_heading:
            pos = html.find('Executive Summary') + 20 + next_heading.start()
            html = html[:pos] + '</div></div>' + html[pos:]
        else:
            html += '</div></div>'
    
    # Style the Trading Plan section similarly
    html = re.sub(r'<h[1-6]\s+class="[^"]*">\s*Trading Plan\s*</h[1-6]>', 
                 r'<div class="card mb-4"><div class="card-header bg-success text-white font-weight-bold">Trading Plan</div><div class="card-body">', 
                 html, flags=re.IGNORECASE)
    if 'Trading Plan' in html:
        # Find the next heading to close the card
        next_heading = re.search(r'<h[1-6]', html[html.find('Trading Plan') + 20:])
        if next_heading:
            pos = html.find('Trading Plan') + 20 + next_heading.start()
            html = html[:pos] + '</div></div>' + html[pos:]
        else:
            html += '</div></div>'
    
    # Style Spanish summary section if present
    spanish_section_patterns = [
        r'<h[1-6][^>]*>Resumen Ejecutivo.*?</h[1-6]>',
        r'<h[1-6][^>]*>RESUMEN EJECUTIVO.*?</h[1-6]>',
        r'<h[1-6][^>]*>Análisis en Español.*?</h[1-6]>'
    ]
    
    for pattern in spanish_section_patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            spanish_section = match.group(0)
            html = html.replace(
                spanish_section, 
                f'<div class="card mt-5 mb-4"><div class="card-header bg-info text-white font-weight-bold">Resumen En Español</div><div class="card-body">'
            )
            # Close the card at the end of the document or before another major section
            end_pattern = r'<h[1-6][^>]*>(?!Executive Summary|Trading Plan|Resumen)'
            end_match = re.search(end_pattern, html[html.find(spanish_section) + len(spanish_section):], re.IGNORECASE)
            if end_match:
                pos = html.find(spanish_section) + len(spanish_section) + end_match.start()
                html = html[:pos] + '</div></div>' + html[pos:]
            else:
                html += '</div></div>'

    # Add Bootstrap styling to lists
    html = html.replace('<ul>', '<ul class="list-group list-group-flush">')
    html = html.replace('<li>', '<li class="list-group-item">')
    
    # Add highlighting for important numbers and percentages in tables
    table_cell_pattern = re.compile(r'<td>(.*?)</td>')
    
    def highlight_values(match):
        cell_content = match.group(1)
        # Highlight percentages
        cell_content = re.sub(r'(\b(\+|-)?[0-9]+(\.[0-9]+)?%\b)', r'<span class="badge badge-secondary">\1</span>', cell_content)
        # Highlight dollar amounts
        cell_content = re.sub(r'(\$[0-9,]+(\.[0-9]+)?)', r'<span class="badge badge-primary">\1</span>', cell_content)
        return f'<td>{cell_content}</td>'
    
    html = table_cell_pattern.sub(highlight_values, html)
    
    # Fix emphasis (italic) and strong (bold) elements
    # Fix issue with <em> inside <strong> and vice versa
    html = re.sub(r'<strong>(.*?)<em>(.*?)</strong>(.*?)</em>', r'<strong>\1</strong><em><strong>\2\3</strong></em>', html)
    html = re.sub(r'<em>(.*?)<strong>(.*?)</em>(.*?)</strong>', r'<em>\1</em><strong><em>\2\3</em></strong>', html)
    
    # Properly nest lists
    html = re.sub(r'</li>\s*<ul>', r'<ul>', html)
    html = re.sub(r'</ul>\s*</li>', r'</li>', html)
    
    # Format special patterns common in stock analyses
    # Ticker symbols - Convert uppercase text surrounded by parentheses to code format
    html = re.sub(r'\(([A-Z]{1,5})\)', r'(<code>\1</code>)', html)
    
    # Format common financial metrics with semantic HTML
    financial_metrics = ['P/E', 'EPS', 'EBITDA', 'ROE', 'ROI', 'ROA', 'P/B', 'PEG']
    for metric in financial_metrics:
        html = html.replace(metric, f'<span class="financial-metric">{metric}</span>')
    
    # Make emphasis elements a bit more prominent for financial analysis
    html = re.sub(r'<em>(.*?)</em>', r'<em class="financial-emphasis">\1</em>', html)
    
    # Fix list continuation formatting
    html = re.sub(r'(<li>.*?)<p>(.*?)</p>', r'\1 \2', html)
    
    # Ensure nested lists have proper styling
    html = re.sub(r'<ul class="list-group list-group-flush"><li>', r'<ul><li>', html)

    return html

# Function to calculate the DCF intrinsic value
def calculate_dcf(fcf, growth_rate, wacc, terminal_growth_rate, years=5):
    # Ensure WACC is sufficiently greater than terminal growth rate
    if wacc - terminal_growth_rate < 0.01:
        wacc = terminal_growth_rate + 0.01  # Ensure at least 1% difference
    
    discount_factors = [(1 + wacc) ** i for i in range(1, years + 1)]
    discounted_fcf = [fcf[i] / discount_factors[i] for i in range(years)]
    
    # Terminal value calculation
    terminal_value = (fcf[-1] * (1 + terminal_growth_rate)) / (wacc - terminal_growth_rate)
    discounted_terminal_value = terminal_value / ((1 + wacc) ** years)
    
    intrinsic_value = sum(discounted_fcf) + discounted_terminal_value
    return intrinsic_value

# Function to calculate the PE-based valuation
def calculate_pe_valuation(eps, industry_pe=None, growth_rate=None):
    if eps <= 0:
        return None
    
    # If industry PE is provided, use it
    if industry_pe:
        return eps * industry_pe
    
    # If growth rate is provided, use PEG ratio approach (PE = Growth Rate * 2)
    if growth_rate:
        pe_multiple = max(10, min(25, growth_rate * 100 * 2))  # Cap between 10 and 25
        return eps * pe_multiple
    
    # Default to a conservative PE of 15
    return eps * 15

# Function to calculate the book value-based valuation
def calculate_book_value_valuation(book_value_per_share, price_to_book_industry=None, roe=None):
    if book_value_per_share <= 0:
        return None
    
    # If industry P/B is provided, use it
    if price_to_book_industry:
        return book_value_per_share * price_to_book_industry
    
    # If ROE is provided, adjust P/B based on ROE quality
    if roe:
        if roe > 0.15:  # High ROE
            return book_value_per_share * 2.5
        elif roe > 0.10:  # Good ROE
            return book_value_per_share * 1.8
        elif roe > 0.05:  # Average ROE
            return book_value_per_share * 1.2
        else:  # Low ROE
            return book_value_per_share * 0.8
    
    # Default to P/B of 1
    return book_value_per_share

def get_stock_data(stock_symbol):
    # Create a Chrome-impersonating session to avoid rate limits
    session = curl_requests.Session(impersonate="chrome")
    stock = yf.Ticker(stock_symbol, session=session)
    hist = stock.history(period="5y")
    current_price = hist['Close'].iloc[-1]
    
    # Calculate 52-week high and low
    high_52week = hist['High'].max()
    low_52week = hist['Low'].min()
    
    # Calculate 50-day and 200-day moving averages
    ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
    ma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
    
    # Calculate RSI
    delta = hist['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    # Calculate MACD
    exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
    exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Get additional financial information
    info = stock.info
    
    # Calculate real price using multiple valuation methods
    try:
        # Initialize valuation methods results
        valuation_methods = {}
        months = 12
        last_year_dates = hist.index[-months:]
        
        # 1. DCF Valuation Method
        try:
            # Check if cashflow data is available
            if hasattr(stock, 'cashflow') and not stock.cashflow.empty and 'Free Cash Flow' in stock.cashflow.index:
                # Getting the most recent Free Cash Flow (FCF)
                fcf = stock.cashflow.loc['Free Cash Flow']
                most_recent_fcf = fcf.iloc[0] if not fcf.empty else None  # Access using iloc to avoid FutureWarning
            else:
                print(f"Warning: No cashflow data available for {stock_symbol}")
                most_recent_fcf = None
            
            print(f"DEBUG - {stock_symbol} FCF: {most_recent_fcf}")
            
            # Skip DCF if FCF is None, negative, or very small relative to market cap
            if most_recent_fcf is None:
                print(f"Skipping DCF for {stock_symbol} due to missing FCF data")
                fcf_is_negative = False
                fcf_to_market_cap = 0
            else:
                fcf_is_negative = most_recent_fcf < 0
                market_cap = info.get('marketCap', 0)
                fcf_to_market_cap = abs(most_recent_fcf) / market_cap if market_cap > 0 else 0
            
            if most_recent_fcf is not None:
                if fcf_is_negative:
                    print(f"Warning: {stock_symbol} has negative Free Cash Flow: {most_recent_fcf}")
                
                if not fcf_is_negative and fcf_to_market_cap > 0.001:  # FCF is positive and significant
                    # Estimate growth rate based on historical data if available
                    try:
                        historical_fcf = fcf.iloc[:3]  # Get last 3 years of FCF if available
                        if len(historical_fcf) >= 2:
                            annual_growth_rates = []
                            for i in range(len(historical_fcf)-1):
                                if historical_fcf.iloc[i+1] > 0 and historical_fcf.iloc[i] > 0:
                                    annual_rate = (historical_fcf.iloc[i] / historical_fcf.iloc[i+1]) - 1
                                    annual_growth_rates.append(annual_rate)
                            
                            if annual_growth_rates:
                                growth_rate = sum(annual_growth_rates) / len(annual_growth_rates)
                                growth_rate = max(0.02, min(0.15, growth_rate))  # Cap between 2% and 15%
                            else:
                                growth_rate = 0.05  # Default if can't calculate
                        else:
                            growth_rate = 0.05  # Default if not enough historical data
                    except Exception as e:
                        print(f"Error calculating historical growth rate: {e}")
                        growth_rate = 0.05  # Default to 5%
                
                # Calculate WACC using stock's beta
                beta = info.get('beta', 1.0)  # Default to 1.0 if beta is not available
                risk_free_rate = 0.03  # Assuming 3% as risk-free rate
                market_return = 0.08  # Assuming 8% as market return
                wacc = risk_free_rate + beta * (market_return - risk_free_rate)
                
                # Ensure WACC is at least 3% higher than terminal growth rate
                terminal_growth_rate = 0.02
                if wacc - terminal_growth_rate < 0.03:
                    wacc = terminal_growth_rate + 0.03
                
                # Retrieve shares outstanding
                shares_outstanding = info.get('sharesOutstanding', None)
                if shares_outstanding is None:
                    raise ValueError("Shares Outstanding not available")
                
                print(f"DEBUG - {stock_symbol} Shares Outstanding: {shares_outstanding}")
                print(f"DEBUG - {stock_symbol} Growth Rate: {growth_rate:.2f}")
                print(f"DEBUG - {stock_symbol} WACC: {wacc:.2f}")
                
                # Calculate DCF valuation
                intrinsic_values = []
                fair_values = []
                
                for i in range(months):
                    fcf_projection = [most_recent_fcf * (1 + growth_rate) ** j for j in range(5)]
                    intrinsic_value = calculate_dcf(fcf_projection, growth_rate, wacc, terminal_growth_rate, 5)
                    fair_value = intrinsic_value / shares_outstanding
                    
                    if i == 0:
                        print(f"DEBUG - {stock_symbol} DCF Intrinsic Value: {intrinsic_value}")
                        print(f"DEBUG - {stock_symbol} DCF Fair Value per Share: {fair_value:.2f}")
                    
                    intrinsic_values.append(intrinsic_value)
                    fair_values.append(fair_value)
                
                dcf_fair_value = sum(fair_values) / len(fair_values)
                valuation_methods['dcf'] = {
                    'value': dcf_fair_value,
                    'weight': 0.3,  # Weight for DCF method
                    'intrinsic_values': intrinsic_values,
                    'fair_values': fair_values,
                    'dates': last_year_dates
                }
            else:
                print(f"Skipping DCF for {stock_symbol} due to negative or insignificant FCF")
        except Exception as e:
            print(f"Error in DCF valuation: {e}")
        
        # 2. PE-based Valuation Method
        try:
            eps = info.get('trailingEps')
            forward_eps = info.get('forwardEps')
            pe_ratio = info.get('trailingPE')
            forward_pe = info.get('forwardPE')
            earnings_growth = info.get('earningsGrowth', 0.05)
            
            if eps and eps > 0:
                # Use industry average PE or calculate based on growth
                industry_pe = 18  # Default industry PE
                pe_fair_value = calculate_pe_valuation(eps, industry_pe, earnings_growth)
                
                # Also calculate using forward PE if available
                if forward_eps and forward_eps > 0 and forward_pe:
                    forward_pe_fair_value = forward_eps * forward_pe
                    # Blend trailing and forward PE values
                    pe_fair_value = (pe_fair_value * 0.4) + (forward_pe_fair_value * 0.6)
                
                print(f"DEBUG - {stock_symbol} PE Fair Value: {pe_fair_value:.2f}")
                
                valuation_methods['pe'] = {
                    'value': pe_fair_value,
                    'weight': 0.3  # Weight for PE method
                }
        except Exception as e:
            print(f"Error in PE valuation: {e}")
        
        # 3. Book Value-based Valuation Method
        try:
            book_value = info.get('bookValue')
            price_to_book = info.get('priceToBook')
            roe = info.get('returnOnEquity')
            
            if book_value and book_value > 0:
                # Calculate fair value based on book value and ROE
                book_fair_value = calculate_book_value_valuation(book_value, price_to_book, roe)
                print(f"DEBUG - {stock_symbol} Book Value Fair Value: {book_fair_value:.2f}")
                
                valuation_methods['book'] = {
                    'value': book_fair_value,
                    'weight': 0.2  # Weight for book value method
                }
        except Exception as e:
            print(f"Error in Book Value valuation: {e}")
        
        # 4. Dividend Discount Model (if applicable)
        try:
            dividend_yield = info.get('dividendYield')
            dividend_rate = info.get('dividendRate')
            
            if dividend_rate and dividend_rate > 0:
                # Simple Gordon Growth Model
                dividend_growth_rate = 0.03  # Assume 3% dividend growth
                required_return = wacc if 'dcf' in valuation_methods else 0.08  # Use WACC from DCF or default
                
                # Gordon Growth Model: P = D / (r - g)
                dividend_fair_value = dividend_rate / (required_return - dividend_growth_rate)
                print(f"DEBUG - {stock_symbol} Dividend Fair Value: {dividend_fair_value:.2f}")
                
                valuation_methods['dividend'] = {
                    'value': dividend_fair_value,
                    'weight': 0.2  # Weight for dividend method
                }
        except Exception as e:
            print(f"Error in Dividend valuation: {e}")
        
        # 5. Relative Valuation (based on current price as a sanity check)
        try:
            # This acts as a sanity check to prevent extreme valuations
            relative_fair_value = current_price
            
            valuation_methods['relative'] = {
                'value': relative_fair_value,
                'weight': 0.1  # Small weight for current price
            }
        except Exception as e:
            print(f"Error in Relative valuation: {e}")
        
        # Calculate weighted average fair value from all methods
        if valuation_methods:
            total_weight = sum(method['weight'] for method in valuation_methods.values())
            weighted_fair_value = sum(method['value'] * method['weight'] for method in valuation_methods.values()) / total_weight
            
            # Normalize weights if some methods are missing
            if total_weight < 0.9:  # If significant methods are missing
                # Adjust weights of available methods
                for method in valuation_methods.values():
                    method['weight'] = method['weight'] / total_weight
            
            print(f"DEBUG - {stock_symbol} Methods used: {list(valuation_methods.keys())}")
            print(f"DEBUG - {stock_symbol} Weighted Fair Value: {weighted_fair_value:.2f}")
        
        # Only generate chart if we have valuation methods
        if valuation_methods:
            # Generate the real price chart
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Plot stock price
            ax1.plot(hist.index[-months:], hist['Close'][-months:], label='Stock Price', color='green')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price ($)', color='black')
            
            # Plot fair values from different methods
            method_colors = {
                'dcf': 'blue',
                'pe': 'red',
                'book': 'purple',
                'dividend': 'orange',
                'relative': 'gray'
            }
            
            # Add horizontal lines for each valuation method
            for method_name, method_data in valuation_methods.items():
                if method_name == 'dcf' and 'dates' in method_data:
                    # For DCF we have time series data
                    ax1.plot(method_data['dates'], method_data['fair_values'], 
                             label=f'DCF Fair Value', linestyle='--', color=method_colors[method_name])
                else:
                    # For other methods we have a single value
                    ax1.axhline(y=method_data['value'], linestyle='--', 
                                color=method_colors[method_name], alpha=0.7,
                                label=f'{method_name.upper()} Fair Value: ${method_data["value"]:.2f}')
            
            # Add weighted average fair value
            ax1.axhline(y=weighted_fair_value, linestyle='-', color='black', linewidth=2,
                        label=f'Weighted Fair Value: ${weighted_fair_value:.2f}')
            
            plt.title(f'{stock_symbol} Stock Price vs Fair Value Estimates (Last Year)')
            fig.tight_layout()  # To avoid label overlaps
            ax1.legend(loc='best')
            
            # Save plot to a BytesIO object
            real_price_chart = io.BytesIO()
            plt.savefig(real_price_chart, format='png')
            real_price_chart.seek(0)
            plt.close()  # Ensure the plot is closed to avoid warnings
        else:
            # Create a simple chart with just the stock price if no valuation methods worked
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(hist.index[-months:], hist['Close'][-months:], label='Stock Price', color='green')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            plt.title(f'{stock_symbol} Stock Price (Last Year)')
            ax.legend(loc='best')
            
            # Save plot to a BytesIO object
            real_price_chart = io.BytesIO()
            plt.savefig(real_price_chart, format='png')
            real_price_chart.seek(0)
            plt.close()  # Ensure the plot is closed to avoid warnings
        
        # Calculate the price difference percentage if we have valuation methods
        if valuation_methods:
            price_difference = ((current_price - weighted_fair_value) / weighted_fair_value) * 100
            
            # Print final values for debugging
            print(f"DEBUG - {stock_symbol} Weighted Fair Value: ${weighted_fair_value:.2f}")
            print(f"DEBUG - {stock_symbol} Current Price: ${current_price:.2f}")
            print(f"DEBUG - {stock_symbol} Price Difference: {price_difference:.2f}%")
        else:
            # If no valuation methods worked, set default values
            weighted_fair_value = current_price  # Default to current price
            price_difference = 0.0  # No difference
            print(f"WARNING - No valuation methods worked for {stock_symbol}, using current price as fair value")
        
        # Store individual method valuations for display
        method_valuations = {}
        if valuation_methods:
            for method_name, method_data in valuation_methods.items():
                method_valuations[method_name] = {
                    'value': method_data['value'],
                    'weight': method_data['weight']
                }
        
        real_price_data = {
            "method_valuations": method_valuations,
            "weighted_fair_value": weighted_fair_value,
            "price_difference": price_difference,
            "real_price_chart": real_price_chart,
            "has_valuation_methods": len(valuation_methods) > 0
        }
    except Exception as e:
        print(f"Error calculating real price: {e}")
        # Create a minimal real_price_data with just the stock price chart
        try:
            # Create a simple chart with just the stock price
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(hist.index[-12:], hist['Close'][-12:], label='Stock Price', color='green')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            plt.title(f'{stock_symbol} Stock Price (Last Year)')
            ax.legend(loc='best')
            
            # Save plot to a BytesIO object
            real_price_chart = io.BytesIO()
            plt.savefig(real_price_chart, format='png')
            real_price_chart.seek(0)
            plt.close()  # Ensure the plot is closed to avoid warnings
            
            real_price_data = {
                "method_valuations": {},
                "weighted_fair_value": current_price,  # Default to current price
                "price_difference": 0.0,  # No difference
                "real_price_chart": real_price_chart,
                "has_valuation_methods": False,
                "error_message": str(e)
            }
        except Exception as chart_error:
            print(f"Error creating fallback chart: {chart_error}")
            real_price_data = None
    
    return {
        "current_price": current_price,
        "high_52week": high_52week,
        "low_52week": low_52week,
        "ma_50": ma_50,
        "ma_200": ma_200,
        "rsi": rsi,
        "macd": macd.iloc[-1],
        "macd_signal": signal.iloc[-1],
        "volume": hist['Volume'].iloc[-1],
        "avg_volume": hist['Volume'].mean(),
        "market_cap": info.get('marketCap'),
        "pe_ratio": info.get('trailingPE'),
        "forward_pe": info.get('forwardPE'),
        "peg_ratio": info.get('pegRatio'),
        "beta": info.get('beta'),
        "dividend_yield": info.get('dividendYield'),
        "earnings_growth": info.get('earningsGrowth'),
        "revenue_growth": info.get('revenueGrowth'),
        "profit_margins": info.get('profitMargins'),
        "debt_to_equity": info.get('debtToEquity'),
        "free_cash_flow": info.get('freeCashflow'),
        "book_value": info.get('bookValue'),
        "price_to_book": info.get('priceToBook'),
        "real_price_data": real_price_data
    }

def get_stock_sentiment(ticker_symbol):
    """
    Analyze sentiment from news articles about the stock
    Returns a dictionary with sentiment scores and recent news
    """
    try:
        # Create a Chrome-impersonating session for Yahoo Finance
        chrome_session = curl_requests.Session(impersonate="chrome")
        
        # No longer using requests_html HTMLSession which causes asyncio errors
        # Instead, just fetch news directly from Yahoo Finance using curl_cffi
        url = f"https://finance.yahoo.com/quote/{ticker_symbol}/news"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        r = chrome_session.get(url, headers=headers, timeout=10)
        
        # Process the response with BeautifulSoup instead of requests_html
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # Extract news headlines and links - adjust selector for Yahoo Finance HTML structure
        news_elements = soup.select('h3.Mb\\(5px\\)')
        if not news_elements:
            # Fallback selectors if the first one doesn't work
            news_elements = soup.select('h3')
            
        # Process up to 10 most recent news items
        news_data = []
        overall_sentiment = 0
        
        for i, headline_elem in enumerate(news_elements[:10]):
            try:
                headline = headline_elem.text
                # Perform sentiment analysis using TextBlob
                analysis = TextBlob(headline)
                sentiment_score = analysis.sentiment.polarity
                
                news_data.append({
                    'headline': headline,
                    'sentiment': sentiment_score,
                    'sentiment_label': 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'
                })
                
                overall_sentiment += sentiment_score
            except Exception as e:
                print(f"Error processing individual news item: {e}")
        
        # Calculate the average sentiment if we have news
        if news_data:
            avg_sentiment = overall_sentiment / len(news_data)
            sentiment_label = 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
        else:
            avg_sentiment = 0
            sentiment_label = 'Neutral'
            
        # Try to get additional market buzz from Finviz
        try:
            finviz_url = f"https://finviz.com/quote.ashx?t={ticker_symbol}"
            finviz_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            finviz_response = curl_requests.get(finviz_url, headers=finviz_headers, timeout=10)
            
            if finviz_response.status_code == 200:
                finviz_soup = BeautifulSoup(finviz_response.text, 'html.parser')
                news_table = finviz_soup.find('table', {'class': 'fullview-news-outer'})
                
                if news_table:
                    rows = news_table.findAll('tr')
                    finviz_news = []
                    
                    for row in rows[:10]:  # Get up to 10 recent news
                        try:
                            date_td = row.find('td', {'align': 'right'})
                            headline_td = row.find('td', {'align': 'left'})
                            
                            if date_td and headline_td:
                                date = date_td.text.strip()
                                headline = headline_td.text.strip()
                                
                                # Perform sentiment analysis
                                analysis = TextBlob(headline)
                                sentiment_score = analysis.sentiment.polarity
                                
                                finviz_news.append({
                                    'date': date,
                                    'headline': headline,
                                    'sentiment': sentiment_score,
                                    'sentiment_label': 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'
                                })
                        except Exception as e:
                            print(f"Error processing Finviz news item: {e}")
                    
                    # Add Finviz news to our news data
                    if finviz_news:
                        # Recalculate overall sentiment with finviz news included
                        overall_sentiment += sum(item['sentiment'] for item in finviz_news)
                        news_data.extend(finviz_news)
                        avg_sentiment = overall_sentiment / len(news_data)
                        sentiment_label = 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
        except Exception as e:
            print(f"Error processing Finviz data: {e}")
        
        # Return structured sentiment data
        return {
            'success': True,
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'news': news_data,
            'news_count': len(news_data)
        }
    
    except Exception as e:
        print(f"Error in get_stock_sentiment: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'sentiment_score': 0,
            'sentiment_label': 'Neutral',
            'news': [],
            'news_count': 0
        }

def get_sector_comparison(ticker_symbol):
    """
    Compare the stock with its sector and industry peers
    Returns a dictionary with comparison metrics
    """
    try:
        # Create a Chrome-impersonating session
        session = curl_requests.Session(impersonate="chrome")
        
        # Get the stock info using yahooquery for more reliable sector/industry data
        ticker = Ticker(ticker_symbol)
        ticker_info = ticker.asset_profile
        
        if ticker_symbol not in ticker_info:
            return {'success': False, 'error': 'Could not retrieve ticker info'}
        
        sector = ticker_info[ticker_symbol].get('sector')
        industry = ticker_info[ticker_symbol].get('industry')
        
        if not sector or not industry:
            # Try to get from yfinance as a fallback
            yf_ticker = yf.Ticker(ticker_symbol, session=session)
            yf_info = yf_ticker.info
            sector = yf_info.get('sector', 'Unknown')
            industry = yf_info.get('industry', 'Unknown')
            
        # Get stock fundamentals
        stock = yf.Ticker(ticker_symbol, session=session)
        stock_info = stock.info
        
        # Get peers from Yahoo Finance
        peers = []
        try:
            peers_raw = ticker.recommendations.get(ticker_symbol, {}).get('recommendedSymbols', [])
            if peers_raw:
                peers = [peer.get('symbol') for peer in peers_raw if peer.get('symbol') != ticker_symbol][:5]
        except Exception as e:
            print(f"Error getting peers via yahooquery: {e}")
            
        # If no peers found, use industry search on Finviz as fallback
        if not peers and industry != 'Unknown':
            try:
                # Create a cleaner industry string for the URL
                industry_url = industry.lower().replace(' ', '')
                finviz_url = f"https://finviz.com/screener.ashx?v=111&f=ind_{industry_url}"
                finviz_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                finviz_response = requests.get(finviz_url, headers=finviz_headers, timeout=10)
                
                if finviz_response.status_code == 200:
                    finviz_soup = BeautifulSoup(finviz_response.content, 'html.parser')
                    ticker_cells = finviz_soup.find_all('a', {'class': 'screener-link-primary'})
                    
                    for cell in ticker_cells:
                        peer_symbol = cell.text.strip()
                        if peer_symbol != ticker_symbol and peer_symbol not in peers:
                            peers.append(peer_symbol)
                            if len(peers) >= 5:  # Limit to 5 peers
                                break
            except Exception as e:
                print(f"Error getting peers via Finviz: {e}")
        
        # If still no peers, add some large companies from the same sector as fallback
        if not peers and sector != 'Unknown':
            # Dictionary of top companies by sector
            sector_leaders = {
                'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
                'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
                'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'],
                'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
                'Industrials': ['HON', 'UNP', 'UPS', 'CAT', 'DE'],
                'Communication Services': ['CMCSA', 'VZ', 'T', 'NFLX', 'DIS'],
                'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
                'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
                'Basic Materials': ['LIN', 'APD', 'ECL', 'SHW', 'NEM'],
                'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
                'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP']
            }
            
            peers = sector_leaders.get(sector, [])
            if ticker_symbol in peers:
                peers.remove(ticker_symbol)
        
        # Analyze each peer
        peer_data = []
        
        def get_peer_metrics(peer):
            try:
                # Create a Chrome-impersonating session for each peer request
                session = curl_requests.Session(impersonate="chrome")
                peer_ticker = yf.Ticker(peer, session=session)
                peer_info = peer_ticker.info
                
                # Extract key metrics
                market_cap = peer_info.get('marketCap', 0)
                pe_ratio = peer_info.get('trailingPE', None)
                forward_pe = peer_info.get('forwardPE', None)
                price_to_book = peer_info.get('priceToBook', None)
                profit_margins = peer_info.get('profitMargins', None)
                return_on_equity = peer_info.get('returnOnEquity', None)
                dividend_yield = peer_info.get('dividendYield', None)
                beta = peer_info.get('beta', None)
                fifty_two_week_change = peer_info.get('52WeekChange', None)
                
                # Get current price
                hist = peer_ticker.history(period="1d")
                current_price = hist['Close'].iloc[-1] if not hist.empty else None
                
                # Build the peer data dictionary
                return {
                    'symbol': peer,
                    'name': peer_info.get('shortName', peer),
                    'market_cap': market_cap,
                    'current_price': current_price,
                    'pe_ratio': pe_ratio,
                    'forward_pe': forward_pe,
                    'price_to_book': price_to_book,
                    'profit_margins': profit_margins,
                    'return_on_equity': return_on_equity,
                    'dividend_yield': dividend_yield,
                    'beta': beta,
                    'fifty_two_week_change': fifty_two_week_change
                }
            except Exception as e:
                print(f"Error getting data for peer {peer}: {e}")
                return {
                    'symbol': peer,
                    'name': peer,
                    'error': str(e)
                }
        
        # Use ThreadPoolExecutor to get peer data in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            peer_data = list(executor.map(get_peer_metrics, peers))
        
        # Filter out peers with errors
        peer_data = [p for p in peer_data if 'error' not in p]
        
        # Calculate key metrics for comparison
        # Add the target stock to the peer list for consistent calculations
        target_stock_data = {
            'symbol': ticker_symbol,
            'name': stock_info.get('shortName', ticker_symbol),
            'market_cap': stock_info.get('marketCap', 0),
            'current_price': stock_info.get('currentPrice', None),
            'pe_ratio': stock_info.get('trailingPE', None),
            'forward_pe': stock_info.get('forwardPE', None),
            'price_to_book': stock_info.get('priceToBook', None),
            'profit_margins': stock_info.get('profitMargins', None),
            'return_on_equity': stock_info.get('returnOnEquity', None),
            'dividend_yield': stock_info.get('dividendYield', None),
            'beta': stock_info.get('beta', None),
            'fifty_two_week_change': stock_info.get('52WeekChange', None)
        }
        
        # Combine target stock and peers for statistical calculations
        all_stocks = [target_stock_data] + peer_data
        
        # Create comparison metrics
        comparison_metrics = {}
        for metric in ['pe_ratio', 'forward_pe', 'price_to_book', 'profit_margins', 
                       'return_on_equity', 'dividend_yield', 'beta']:
            valid_values = [stock[metric] for stock in all_stocks if stock[metric] is not None]
            if valid_values:
                metric_mean = np.mean(valid_values)
                metric_median = np.median(valid_values)
                metric_min = min(valid_values)
                metric_max = max(valid_values)
                metric_std = np.std(valid_values)
                
                target_value = target_stock_data[metric]
                if target_value is not None:
                    percentile = stats.percentileofscore(valid_values, target_value)
                    z_score = (target_value - metric_mean) / metric_std if metric_std > 0 else 0
                    
                    comparison_metrics[metric] = {
                        'value': target_value,
                        'mean': metric_mean,
                        'median': metric_median,
                        'min': metric_min,
                        'max': metric_max,
                        'std': metric_std,
                        'percentile': percentile,
                        'z_score': z_score,
                        'relative_position': 'above_average' if target_value > metric_mean else 'below_average'
                    }
        
        # Generate the comparison charts data
        chart_data = {}
        
        # Create bar chart data for PE ratio comparison
        if 'pe_ratio' in comparison_metrics:
            chart_data['pe_ratio'] = {
                'labels': [stock['symbol'] for stock in all_stocks if stock['pe_ratio'] is not None],
                'values': [stock['pe_ratio'] for stock in all_stocks if stock['pe_ratio'] is not None]
            }
        
        # Create bar chart data for Price/Book comparison
        if 'price_to_book' in comparison_metrics:
            chart_data['price_to_book'] = {
                'labels': [stock['symbol'] for stock in all_stocks if stock['price_to_book'] is not None],
                'values': [stock['price_to_book'] for stock in all_stocks if stock['price_to_book'] is not None]
            }
        
        # Create bar chart data for Return on Equity comparison
        if 'return_on_equity' in comparison_metrics:
            chart_data['return_on_equity'] = {
                'labels': [stock['symbol'] for stock in all_stocks if stock['return_on_equity'] is not None],
                'values': [stock['return_on_equity'] for stock in all_stocks if stock['return_on_equity'] is not None]
            }
        
        return {
            'success': True,
            'sector': sector,
            'industry': industry,
            'peers': peer_data,
            'target_stock': target_stock_data,
            'comparison_metrics': comparison_metrics,
            'chart_data': chart_data
        }
    
    except Exception as e:
        print(f"Error in get_sector_comparison: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'sector': 'Unknown',
            'industry': 'Unknown',
            'peers': [],
            'comparison_metrics': {}
        }

def get_advanced_technical_indicators(ticker_symbol):
    """
    Calculate advanced technical indicators for the stock
    Returns a dictionary with various technical indicators and signals
    """
    try:
        # Create a Chrome-impersonating session
        session = curl_requests.Session(impersonate="chrome")
        
        # Get historical data with timeouts
        try:
            stock = yf.Ticker(ticker_symbol, session=session)
            hist = stock.history(period="1y", timeout=10)  # Add timeout parameter
        except Exception as e:
            print(f"Error getting history: {e}")
            # Try with shorter timeframe if full year fails
            try:
                hist = stock.history(period="6mo", timeout=5) 
            except:
                # Last resort - very small timeframe
                hist = stock.history(period="1mo", timeout=5)
        
        if hist.empty:
            return {'success': False, 'error': 'No historical data available'}
        
        # Current price
        current_price = hist['Close'].iloc[-1]
        
        # Calculate basic indicators - with timeouts and optimizations
        # Moving Averages - limit periods to reduce calculation time
        ma_periods = [5, 20, 50, 200]  # Reduced number of periods
        moving_averages = {}
        
        for period in ma_periods:
            if len(hist) >= period:
                ma = hist['Close'].rolling(window=period).mean()
                moving_averages[f'MA_{period}'] = ma.iloc[-1]
        
        # Exponential Moving Averages - limit periods
        ema_periods = [12, 26, 50]  # Reduced number of periods
        emas = {}
        
        for period in ema_periods:
            if len(hist) >= period:
                ema = hist['Close'].ewm(span=period, adjust=False).mean()
                emas[f'EMA_{period}'] = ema.iloc[-1]
        
        # Bollinger Bands
        bollinger_period = 20
        bollinger_bands = None
        if len(hist) >= bollinger_period:
            ma20 = hist['Close'].rolling(window=bollinger_period).mean()
            std20 = hist['Close'].rolling(window=bollinger_period).std()
            
            bollinger_upper = ma20 + (std20 * 2)
            bollinger_lower = ma20 - (std20 * 2)
            
            bollinger_bands = {
                'middle': ma20.iloc[-1],
                'upper': bollinger_upper.iloc[-1],
                'lower': bollinger_lower.iloc[-1],
                'width': (bollinger_upper.iloc[-1] - bollinger_lower.iloc[-1]) / ma20.iloc[-1]
            }
        
        # RSI Calculation
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # Limit RSI periods
        rsi_periods = [14]  # Just use standard 14-day RSI
        rsi_values = {}
        
        for period in rsi_periods:
            if len(hist) >= period:
                rsi = calculate_rsi(hist['Close'], period)
                rsi_values[f'RSI_{period}'] = rsi.iloc[-1]
        
        # MACD
        macd = None
        if len(hist) >= 26:
            ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
            ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_histogram = macd_line - signal_line
            
            macd = {
                'macd_line': macd_line.iloc[-1],
                'signal_line': signal_line.iloc[-1],
                'histogram': macd_histogram.iloc[-1]
            }
        
        # Stochastic Oscillator
        stochastic = None
        if len(hist) >= 14:
            low_14 = hist['Low'].rolling(window=14).min()
            high_14 = hist['High'].rolling(window=14).max()
            k = 100 * ((hist['Close'] - low_14) / (high_14 - low_14))
            d = k.rolling(window=3).mean()
            
            stochastic = {
                'k': k.iloc[-1],
                'k_3_days': k.iloc[-3:].tolist() if len(k) >= 3 else [],
                'd': d.iloc[-1],
                'd_3_days': d.iloc[-3:].tolist() if len(d) >= 3 else []
            }
        
        # Only do minimal ADX calculation - this is one of the slow parts
        adx = None
        if len(hist) >= 28:  # Need more periods to calculate ADX reliably
            try:
                # Simplified ADX calculation
                tr1 = hist['High'] - hist['Low']
                tr2 = abs(hist['High'] - hist['Close'].shift(1))
                tr3 = abs(hist['Low'] - hist['Close'].shift(1))
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = tr.rolling(window=14).mean()
                
                plus_dm = hist['High'].diff()
                minus_dm = hist['Low'].diff().multiply(-1)
                
                plus_dm = plus_dm.mask((plus_dm <= 0) | (plus_dm <= minus_dm), 0)
                minus_dm = minus_dm.mask((minus_dm <= 0) | (minus_dm <= plus_dm), 0)
                
                plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
                
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 0.0001)
                adx_value = dx.rolling(window=14).mean()
                
                adx = {
                    'adx': adx_value.iloc[-1] if not pd.isna(adx_value.iloc[-1]) else 0,
                    'plus_di': plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0,
                    'minus_di': minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0
                }
            except Exception as e:
                print(f"Error calculating ADX: {e}")
                # Skip ADX if calculation fails
                adx = {'adx': 0, 'plus_di': 0, 'minus_di': 0}
        
        # Skip Aroon calculation - it's slow and rarely used
        aroon = None
        
        # Relative Volume - simple calculation
        avg_volume = hist['Volume'].rolling(window=20).mean().iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        relative_volume = current_volume / avg_volume if avg_volume > 0 else 0
        
        # Skip OBV and MFI calculations - they're computation heavy
        money_flow_index = None
        
        # Skip Ichimoku Cloud - very computation heavy
        ichimoku = None
        
        # Fibonacci Retracement Levels - simpler calculation
        # Calculate based on recent high and low instead of 52-week
        recent_high = hist['High'][-60:].max()  # 3 month high
        recent_low = hist['Low'][-60:].min()    # 3 month low
        price_range = recent_high - recent_low
        
        fibonacci_levels = {
            '0.0': recent_low,
            '0.236': recent_low + 0.236 * price_range,
            '0.382': recent_low + 0.382 * price_range,
            '0.5': recent_low + 0.5 * price_range,
            '0.618': recent_low + 0.618 * price_range,
            '0.786': recent_low + 0.786 * price_range,
            '1.0': recent_high
        }
        
        # Find closest Fibonacci level
        closest_level = min(fibonacci_levels.items(), key=lambda x: abs(float(x[1]) - current_price))
        
        # Simplify Support and Resistance Levels calculation
        # Use fixed number of points rather than complex algorithm
        recent_hist = hist.tail(20)  # last month of trading
        
        # Just pick a few recent highs and lows
        highs = sorted(recent_hist['High'].nlargest(3).tolist())
        lows = sorted(recent_hist['Low'].nsmallest(3).tolist())
        
        # Filter resistance levels above current price
        resistance_levels = [price for price in highs if price > current_price]
        # Filter support levels below current price
        support_levels = [price for price in lows if price < current_price]
        
        # Generate simplified trading signals
        signals = []
        
        # MA Crossovers - only do if we have the needed MAs
        if 'MA_50' in moving_averages and 'MA_200' in moving_averages:
            if moving_averages['MA_50'] > moving_averages['MA_200']:
                signals.append({
                    'indicator': 'MA Crossover',
                    'signal': 'bullish',
                    'description': 'Golden Cross: 50-day MA above 200-day MA'
                })
            elif moving_averages['MA_50'] < moving_averages['MA_200']:
                signals.append({
                    'indicator': 'MA Crossover',
                    'signal': 'bearish',
                    'description': 'Death Cross: 50-day MA below 200-day MA'
                })
        
        # RSI Signal
        if 'RSI_14' in rsi_values:
            rsi14 = rsi_values['RSI_14']
            if rsi14 > 70:
                signals.append({
                    'indicator': 'RSI',
                    'signal': 'bearish',
                    'description': f'Overbought RSI: {rsi14:.2f}'
                })
            elif rsi14 < 30:
                signals.append({
                    'indicator': 'RSI',
                    'signal': 'bullish',
                    'description': f'Oversold RSI: {rsi14:.2f}'
                })
        
        # MACD Signal
        if macd:
            if macd['macd_line'] > macd['signal_line']:
                signals.append({
                    'indicator': 'MACD',
                    'signal': 'bullish',
                    'description': 'MACD Line above Signal Line'
                })
            elif macd['macd_line'] < macd['signal_line']:
                signals.append({
                    'indicator': 'MACD',
                    'signal': 'bearish',
                    'description': 'MACD Line below Signal Line'
                })
        
        # Bollinger Bands Signal
        if bollinger_bands:
            if current_price > bollinger_bands['upper']:
                signals.append({
                    'indicator': 'Bollinger Bands',
                    'signal': 'bearish',
                    'description': 'Price above Upper Bollinger Band'
                })
            elif current_price < bollinger_bands['lower']:
                signals.append({
                    'indicator': 'Bollinger Bands',
                    'signal': 'bullish',
                    'description': 'Price below Lower Bollinger Band'
                })
        
        # Only include Stochastic signal if we have data
        if stochastic and stochastic['k'] is not None and stochastic['d'] is not None:
            if stochastic['k'] > 80 and stochastic['d'] > 80:
                signals.append({
                    'indicator': 'Stochastic',
                    'signal': 'bearish',
                    'description': 'Stochastic Overbought'
                })
            elif stochastic['k'] < 20 and stochastic['d'] < 20:
                signals.append({
                    'indicator': 'Stochastic',
                    'signal': 'bullish',
                    'description': 'Stochastic Oversold'
                })
        
        # Get overall signal
        bullish_signals = len([s for s in signals if s['signal'] == 'bullish'])
        bearish_signals = len([s for s in signals if s['signal'] == 'bearish'])
        
        if bullish_signals > bearish_signals:
            overall_signal = {
                'signal': 'bullish',
                'strength': f'{bullish_signals}/{len(signals)}',
                'description': f'{bullish_signals} bullish signals vs {bearish_signals} bearish signals'
            }
        elif bearish_signals > bullish_signals:
            overall_signal = {
                'signal': 'bearish',
                'strength': f'{bearish_signals}/{len(signals)}',
                'description': f'{bearish_signals} bearish signals vs {bullish_signals} bullish signals'
            }
        else:
            overall_signal = {
                'signal': 'neutral',
                'strength': '50/50',
                'description': 'Equal bullish and bearish signals'
            }
        
        # Generate simplified chart data for visualization
        try:
            # Get only what we need for the chart - last 30 days
            chart_dates = hist.index[-30:].strftime('%Y-%m-%d').tolist()
            chart_prices = hist['Close'][-30:].tolist()
            chart_volumes = hist['Volume'][-30:].tolist()
            
            chart_data = {
                'dates': chart_dates,
                'prices': chart_prices,
                'volumes': chart_volumes,
                'ma20': hist['Close'].rolling(window=20).mean()[-30:].tolist() if len(hist) >= 20 else [],
                'ma50': hist['Close'].rolling(window=50).mean()[-30:].tolist() if len(hist) >= 50 else []
            }
            
            # Only add Bollinger bands if calculated
            if bollinger_bands:
                recent_hist = hist[-30:]
                ma20 = recent_hist['Close'].rolling(window=20).mean()
                std20 = recent_hist['Close'].rolling(window=20).std()
                upper_band = (ma20 + (std20 * 2)).tolist()
                lower_band = (ma20 - (std20 * 2)).tolist()
                
                chart_data['upper_band'] = upper_band
                chart_data['lower_band'] = lower_band
            else:
                chart_data['upper_band'] = []
                chart_data['lower_band'] = []
        except Exception as e:
            print(f"Error creating chart data: {e}")
            # Fallback minimal chart data
            chart_data = {
                'dates': hist.index[-10:].strftime('%Y-%m-%d').tolist(),
                'prices': hist['Close'][-10:].tolist(),
                'volumes': hist['Volume'][-10:].tolist(),
                'ma20': [],
                'ma50': [],
                'upper_band': [],
                'lower_band': []
            }
        
        return {
            'success': True,
            'current_price': current_price,
            'moving_averages': moving_averages,
            'emas': emas,
            'bollinger_bands': bollinger_bands,
            'rsi': rsi_values,
            'macd': macd,
            'stochastic': stochastic,
            'adx': adx,
            'aroon': aroon,
            'relative_volume': relative_volume,
            'money_flow_index': money_flow_index,
            'ichimoku': ichimoku,
            'fibonacci_levels': fibonacci_levels,
            'closest_fibonacci_level': closest_level,
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'signals': signals,
            'overall_signal': overall_signal,
            'chart_data': chart_data
        }
    
    except Exception as e:
        print(f"Error in get_advanced_technical_indicators: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }

# Add a response caching system
def save_response_to_cache(stock_symbol, html_content):
    """Save the GPT response to a file cache"""
    cache_dir = os.path.join(os.getcwd(), "response_cache")
    
    # Create the cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Create a filename based on the stock symbol and timestamp
    filename = f"{stock_symbol}_{hashlib.md5(stock_symbol.encode()).hexdigest()}.html"
    file_path = os.path.join(cache_dir, filename)
    
    # Write the HTML content to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    # Store just the filename in the session
    session['response_cache_file'] = filename
    
    return filename

def load_response_from_cache(stock_symbol=None, filename=None):
    """Load the GPT response from the file cache"""
    cache_dir = os.path.join(os.getcwd(), "response_cache")
    
    # If a specific filename is provided, use that
    if filename:
        file_path = os.path.join(cache_dir, filename)
    # Otherwise check if there's a file for this stock
    elif stock_symbol:
        # Create the filename pattern
        pattern = f"{stock_symbol}_"
        
        # List files in the cache directory
        if os.path.exists(cache_dir):
            files = os.listdir(cache_dir)
            # Find the most recent file for this stock
            matching_files = [f for f in files if f.startswith(pattern)]
            
            if matching_files:
                # Sort by creation time (newest first)
                matching_files.sort(key=lambda x: os.path.getctime(os.path.join(cache_dir, x)), reverse=True)
                file_path = os.path.join(cache_dir, matching_files[0])
            else:
                return None
        else:
            return None
    else:
        return None
    
    # Read the HTML content from the file if it exists
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    
    return None

@app.route('/', methods=['GET', 'POST'])
def index_scrap():
    if request.method == 'POST':
        stock = request.form['stock']
        routes = ["cheat-sheet", "technical-analysis", "performance", "analyst-ratings"]
        #routes = ["cheat-sheet"]
        scraped_content = {}

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            for route in routes:
                url = f"https://www.barchart.com/stocks/quotes/{stock}/{route}"
                # Use curl_cffi requests for Barchart scraping
                response = curl_requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    main_content_div = soup.find(id="main-content-column")
                    if main_content_div:
                        scraped_content[route] = main_content_div.get_text(strip=True)
                    else:
                        scraped_content[route] = "[main-content-column not found]"
                else:
                    scraped_content[route] = f"[Failed to fetch: HTTP {response.status_code}]"

            # Get the standard stock data
            stock_data = get_stock_data(stock)
            
            # Get advanced analytical data - with error handling and timeouts
            start_time = time.time()
            print(f"[INFO] Fetching additional analysis for {stock}")
            
            # Use a smaller thread pool and longer timeouts
            # Run each analysis individually rather than in parallel
            try:
                # First get sentiment data
                sentiment_data = get_stock_sentiment(stock)
                print(f"[INFO] Sentiment analysis completed in {time.time() - start_time:.2f} seconds")
                
                # Then get technical indicators with timeout
                technical_indicators = get_advanced_technical_indicators(stock)
                print(f"[INFO] Technical analysis completed in {time.time() - start_time:.2f} seconds")
                
                # Finally get sector comparison 
                sector_comparison = get_sector_comparison(stock)
                print(f"[INFO] Sector analysis completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                print(f"[ERROR] Error during analysis: {e}")
                # Provide default data for any missing analysis
                if 'sentiment_data' not in locals():
                    sentiment_data = {'success': False, 'error': str(e), 'sentiment_label': 'Neutral', 'news': []}
                if 'technical_indicators' not in locals():
                    technical_indicators = {'success': False, 'error': str(e)}
                if 'sector_comparison' not in locals():
                    sector_comparison = {'success': False, 'error': str(e)}
            
            print(f"[INFO] Additional analysis fetched in {time.time() - start_time:.2f} seconds")
            
            # Add the new data to stock_data for the template
            stock_data['sentiment'] = sentiment_data
            stock_data['sector_comparison'] = sector_comparison
            stock_data['technical_indicators'] = technical_indicators
            
            # Add comprehensive valuation information to the prompt if available
            valuation_info = ""
            if stock_data['real_price_data']:
                # Start with the weighted fair value
                valuation_info = f"""
                Valuation Analysis:
                Weighted Fair Value: ${stock_data['real_price_data']['weighted_fair_value']:.2f}
                Current Price vs Fair Value: {stock_data['real_price_data']['price_difference']:.2f}% {'overvalued' if stock_data['real_price_data']['price_difference'] > 0 else 'undervalued'}
                """
                
                # Add individual valuation methods if available
                if stock_data['real_price_data']['has_valuation_methods']:
                    valuation_info += "\nIndividual Valuation Methods:\n"
                    
                    for method_name, method_data in stock_data['real_price_data']['method_valuations'].items():
                        method_description = {
                            'dcf': 'Discounted Cash Flow (based on future free cash flows)',
                            'pe': 'Price-to-Earnings (based on earnings multiples)',
                            'book': 'Book Value (based on company assets)',
                            'dividend': 'Dividend Discount Model (based on dividend yield)',
                            'relative': 'Relative Valuation (based on current market price)'
                        }.get(method_name, method_name.upper())
                        
                        valuation_info += f"- {method_description}: ${method_data['value']:.2f} (Weight: {method_data['weight']*100:.0f}%)\n"
                        
                    # Add explanation of what the valuation means
                    if stock_data['real_price_data']['price_difference'] > 20:
                        valuation_info += "\nThe stock appears significantly overvalued based on multiple valuation methods.\n"
                    elif stock_data['real_price_data']['price_difference'] > 0:
                        valuation_info += "\nThe stock appears somewhat overvalued based on multiple valuation methods.\n"
                    elif stock_data['real_price_data']['price_difference'] < -20:
                        valuation_info += "\nThe stock appears significantly undervalued based on multiple valuation methods.\n"
                    elif stock_data['real_price_data']['price_difference'] < 0:
                        valuation_info += "\nThe stock appears somewhat undervalued based on multiple valuation methods.\n"
                    else:
                        valuation_info += "\nThe stock appears fairly valued based on multiple valuation methods.\n"

            # Add sentiment analysis to prompt
            sentiment_info = ""
            if sentiment_data['success']:
                sentiment_info = f"""
                Market Sentiment Analysis:
                Overall Sentiment: {sentiment_data['sentiment_label'].upper()} (Score: {sentiment_data['sentiment_score']:.2f})
                News Count: {sentiment_data['news_count']}
                """
                
                # Add recent news headlines - limit to 3 to reduce prompt size
                if sentiment_data['news']:
                    sentiment_info += "\nRecent News Headlines:\n"
                    for i, news in enumerate(sentiment_data['news'][:3]):  # Only include 3 news items
                        headline = news.get('headline', '')
                        sentiment = news.get('sentiment_label', '')
                        sentiment_info += f"- {headline} ({sentiment})\n"

            # Add technical indicators to prompt
            technical_info = ""
            if technical_indicators['success']:
                technical_info = """
                Advanced Technical Analysis:
                """
                
                # Add support and resistance levels
                if technical_indicators['support_levels']:
                    # Limit to top 3 support levels to reduce prompt size
                    support_str = ', '.join([f"${level:.2f}" for level in technical_indicators['support_levels'][:3]])
                    technical_info += f"\nSupport Levels: {support_str}"
                
                if technical_indicators['resistance_levels']:
                    # Limit to top 3 resistance levels to reduce prompt size
                    resistance_str = ', '.join([f"${level:.2f}" for level in technical_indicators['resistance_levels'][:3]])
                    technical_info += f"\nResistance Levels: {resistance_str}"
                
                # Add Fibonacci levels
                if technical_indicators['closest_fibonacci_level']:
                    fib_level, fib_value = technical_indicators['closest_fibonacci_level']
                    technical_info += f"\nClosest Fibonacci Level: {fib_level} at ${fib_value:.2f}"
                
                # Add signals summary
                if technical_indicators['signals']:
                    bullish = len([s for s in technical_indicators['signals'] if s['signal'] == 'bullish'])
                    bearish = len([s for s in technical_indicators['signals'] if s['signal'] == 'bearish'])
                    neutral = len([s for s in technical_indicators['signals'] if s['signal'] == 'neutral'])
                    technical_info += f"\nTechnical Signals: {bullish} Bullish, {bearish} Bearish, {neutral} Neutral"
                    
                    # Add individual signals - limit to 6 signals to reduce prompt size
                    if len(technical_indicators['signals']) > 0:
                        technical_info += "\n\nDetailed Technical Signals:"
                        for signal in technical_indicators['signals'][:6]:  # Only include top 6 signals
                            technical_info += f"\n- {signal['indicator']}: {signal['description']} ({signal['signal'].upper()})"
                    
                    if 'overall_signal' in technical_indicators:
                        technical_info += f"\n\nOverall Technical Signal: {technical_indicators['overall_signal']['description']} ({technical_indicators['overall_signal']['signal'].upper()})"

            # Add sector comparison to prompt
            sector_info = ""
            if sector_comparison['success']:
                sector_info = f"""
                Sector and Industry Analysis:
                Sector: {sector_comparison['sector']}
                Industry: {sector_comparison['industry']}
                """
                
                # Add peer comparison - but limited to 2 peers
                if sector_comparison['peers']:
                    sector_info += "\nPeer Comparison:\n"
                    # Show only 2 peers to reduce prompt size
                    for peer in sector_comparison['peers'][:2]:
                        peer_name = peer.get('name', peer.get('symbol', 'N/A'))
                        peer_pe = peer.get('pe_ratio', 'N/A')
                        peer_price = peer.get('current_price', 'N/A')
                        peer_market_cap = peer.get('market_cap', 0)
                        
                        sector_info += f"- {peer_name} ({peer.get('symbol', 'N/A')}): PE {peer_pe}, Price ${peer_price}, Market Cap ${peer_market_cap:,}\n"
                
                # Add comparison metrics - limit to just 2 key metrics
                if sector_comparison['comparison_metrics']:
                    sector_info += "\nRelative to Industry:\n"
                    # Only include 2 most important metrics
                    metrics_to_show = ['pe_ratio', 'price_to_book']
                    for metric_name in metrics_to_show:
                        if metric_name in sector_comparison['comparison_metrics']:
                            metric = sector_comparison['comparison_metrics'][metric_name]
                            metric_display = {
                                'pe_ratio': 'P/E Ratio',
                                'price_to_book': 'Price to Book'
                            }.get(metric_name, metric_name)
                            
                            sector_info += f"- {metric_display}: {metric['value']:.2f} vs Industry Avg: {metric['mean']:.2f} ({metric['percentile']:.0f}th percentile)\n"

            # Use a smaller version of scraped content to reduce prompt size
            scraped_summary = ""
            if scraped_content:
                # Just take first 500 chars of each part
                for route, content in scraped_content.items():
                    if content and len(content) > 100:  # Only include non-empty content
                        scraped_summary += f"\n{route}: {content[:200]}...\n"

            prompt = f"""
            Analyze the current position in {stock} stock based on the following comprehensive data. Technical analysis and fundamental analysis.

            
            TECHNICAL INDICATORS:
            Current Price: ${stock_data['current_price']:.2f}
            52-Week High: ${stock_data['high_52week']:.2f} (Distance: {((stock_data['high_52week'] - stock_data['current_price']) / stock_data['current_price'] * 100):.2f}%)
            52-Week Low: ${stock_data['low_52week']:.2f} (Distance: {((stock_data['current_price'] - stock_data['low_52week']) / stock_data['low_52week'] * 100):.2f}%)
            50-Day Moving Average: ${stock_data['ma_50']:.2f} (Price vs MA: {((stock_data['current_price'] - stock_data['ma_50']) / stock_data['ma_50'] * 100):.2f}%)
            200-Day Moving Average: ${stock_data['ma_200']:.2f} (Price vs MA: {((stock_data['current_price'] - stock_data['ma_200']) / stock_data['ma_200'] * 100):.2f}%)
            RSI (14-day): {stock_data['rsi']:.2f} {'(Overbought)' if stock_data['rsi'] > 70 else '(Oversold)' if stock_data['rsi'] < 30 else '(Neutral)'}
            MACD: {stock_data['macd']:.2f}
            MACD Signal: {stock_data['macd_signal']:.2f}
            MACD Histogram: {(stock_data['macd'] - stock_data['macd_signal']):.2f} {'(Bullish)' if stock_data['macd'] > stock_data['macd_signal'] else '(Bearish)'}
            
            #Technical Info:
            {technical_info}
            
            #Sentiment Info:
            {sentiment_info}

            #Sector Info:
            {sector_info}
            
            
            VOLUME ANALYSIS:
            Current Volume: {stock_data['volume']:,}
            Average Volume: {stock_data['avg_volume']:.0f}
            Volume Ratio: {(stock_data['volume'] / stock_data['avg_volume']):.2f}x {'(High)' if stock_data['volume'] > stock_data['avg_volume'] * 1.5 else '(Low)' if stock_data['volume'] < stock_data['avg_volume'] * 0.5 else '(Normal)'}

            FUNDAMENTAL ANALYSIS:
            Market Cap: ${stock_data['market_cap']:,}
            P/E Ratio: {stock_data['pe_ratio'] if stock_data['pe_ratio'] is not None else 'N/A'}
            Forward P/E: {stock_data['forward_pe'] if stock_data['forward_pe'] is not None else 'N/A'}
            PEG Ratio: {stock_data['peg_ratio'] if stock_data['peg_ratio'] is not None else 'N/A'}
            Beta: {stock_data['beta'] if stock_data['beta'] is not None else 'N/A'}
            Dividend Yield: {stock_data['dividend_yield'] if stock_data['dividend_yield'] is not None else 'N/A'}
            Earnings Growth: {stock_data['earnings_growth'] if stock_data['earnings_growth'] is not None else 'N/A'}
            Revenue Growth: {stock_data['revenue_growth'] if stock_data['revenue_growth'] is not None else 'N/A'}
            Profit Margins: {stock_data['profit_margins'] if stock_data['profit_margins'] is not None else 'N/A'}
            Debt to Equity: {stock_data['debt_to_equity'] if stock_data['debt_to_equity'] is not None else 'N/A'}
            Free Cash Flow: ${stock_data['free_cash_flow']:,} if stock_data['free_cash_flow'] is not None else 'N/A'
            Book Value: ${stock_data['book_value'] if stock_data['book_value'] is not None else 'N/A'}
            Price to Book: {stock_data['price_to_book'] if stock_data['price_to_book'] is not None else 'N/A'}

            {valuation_info}

            ADDITIONAL MARKET INFORMATION:
            {scraped_summary}

            Based on this comprehensive data, please provide:

            0. SUMMARY OF THE STOCK MARKET AND INDUSTRY
                - What are the main products/services of the company?
                - What is the market size and growth potential?
                - What are the main competitors?
                - What are the main market drivers?
                - What are the main market risks?

            1. VALUATION SUMMARY:
               - Intrinsic value assessment and fair price range
               - Detailed explanation of whether the stock is overvalued or undervalued and by what percentage
               - Comparison of current valuation metrics to industry averages and historical trends
               - Identification of key value drivers and potential catalysts

            2. TECHNICAL ANALYSIS:
               - Current price trend and momentum indicators assessment
               - Key support and resistance levels with specific price points
               - Chart pattern identification and potential breakout/breakdown scenarios
               - Volume analysis and what it indicates about market sentiment
               - Moving average analysis and crossover implications

            3. MARKET SENTIMENT:
               - Analysis of current news sentiment and its impact on stock price
               - Social media and investor sentiment trends
               - Insider trading activity and institutional ownership changes
               - Recent analyst upgrades/downgrades and price target changes
               - How sentiment aligns with technical and fundamental indicators

            4. FUNDAMENTAL OUTLOOK:
               - Assessment of financial health and business model strength
               - Growth prospects and competitive positioning
               - Earnings quality and sustainability analysis
               - Risk factors that could impact future performance
               - Dividend sustainability and capital return outlook (if applicable)

            5. INVESTMENT STRATEGY:
               - Clear buy/sell/hold recommendation with detailed rationale
               - Multiple entry strategies with specific price points
               - Multiple exit strategies including profit targets and time horizons
               - Risk management approach with specific stop-loss levels
               - Position sizing recommendations based on risk/reward profile

            6. TRADING PLAN:
               - Clear buy/sell/hold recommendation with detailed rationale
               - Multiple entry strategies with specific price points
               - Multiple exit strategies including profit targets and time horizons
               - Risk management approach with specific stop-loss levels
               - Position sizing recommendations based on risk/reward profile

            7. CONSIDERATIONS:
               - Market conditions and potential catalysts
               - Risk factors and mitigation strategies
               - Liquidity and market access
               - Transaction costs and fees
               - Time horizon and market volatility
               
            8. EXECUTIVE SUMMARY (as a table or bullet points):
               Entry price: [specific price or price range]
               Trade direction: [Buy/Sell/Hold]
               Take Profit (TP): [multiple levels with percentages]
               Stop Loss (SL): [specific level with percentage]
               Risk/Reward Ratio: [calculated ratio]
               Time Horizon: [short-term/medium-term/long-term]
               Conviction Level: [high/medium/low]

            9. FINAL CONCLUSION: 
               - Summary of the analysis
               - Key takeaways
               - Next steps
               - Additional recommendations

            Please be specific with price levels and percentages throughout your analysis. Consider transaction costs, market volatility, and liquidity in your recommendations.
            
            At the end, write a FULL Executive Summary of the analysis in Spanish. The Spanish summary should be a thorough overview of all major points of your analysis.

            Ensure that:
            - The analysis is based on the data provided and is thorough
            - All sections are detailed and data-driven  
            - Prices, targets, and strategies are quantified  
            - Writing is professional and clear
            - You give a well-reasoned investment recommendation
            - The Spanish summary is comprehensive (500-800 words)
            - Reply in markdown format
            """
            
            print(f"[INFO] Prompt length: {len(prompt)} characters")
            gpt_response_html = chat_with_gpt(prompt)
            print(f"[INFO] Generated analysis HTML length: {len(gpt_response_html)}")

            # Save the response to cache instead of storing directly in session
            cache_filename = save_response_to_cache(stock, gpt_response_html)
            print(f"[INFO] Saved GPT response to cache file: {cache_filename}")
            
            # Store minimal data in session
            session['stock_symbol'] = stock
            session['analysis_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Store in g object for the current request
            g.last_gpt_response = gpt_response_html
            g.last_stock = stock

            return render_template('index_scrap.html', now=datetime.now, stock=stock, scraped_content=scraped_content, gpt_response=gpt_response_html, stock_data=stock_data)

        
        finally:
            pass
    
    return render_template('index_scrap.html',now = datetime.now)

@app.route('/real_price_chart/<stock>')
def real_price_chart(stock):
    try:
        stock_data = get_stock_data(stock)
        if stock_data['real_price_data'] and stock_data['real_price_data']['real_price_chart']:
            return send_file(stock_data['real_price_data']['real_price_chart'], mimetype='image/png')
        else:
            return "Real price chart not available", 404
    except Exception as e:
        print(f"Error serving real price chart: {e}")
        return str(e), 500

def generate_pdf_from_html(html_content, stock_symbol):
    """
    Generate a PDF from HTML content
    Returns the path to the generated PDF file
    """
    try:
        # Create a temporary file for the PDF
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = pdf_file.name
        pdf_file.close()
        
        # Set options for PDF generation
        options = {
            'page-size': 'Letter',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'custom-header': [
                ('Accept-Encoding', 'gzip')
            ],
            'no-outline': None,
            'title': f'Stock Analysis - {stock_symbol}',
            'footer-center': 'StockGPT Analysis Report',
            'footer-right': '[page] of [topage]',
            'footer-font-size': '8',
            'header-center': f'Advanced Stock Analysis - {stock_symbol}',
            'header-font-size': '8',
            'header-spacing': '5'
        }
        
        # Configure wkhtmltopdf path based on operating system
        try:
            # Check if wkhtmltopdf is in PATH
            import subprocess
            try:
                # Try to find wkhtmltopdf in the PATH
                wkhtmltopdf_path = subprocess.check_output(['which', 'wkhtmltopdf']).decode().strip()
                print(f"[INFO] Found wkhtmltopdf at: {wkhtmltopdf_path}")
                config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
            except subprocess.CalledProcessError:
                # Manual detection based on common install locations
                import platform
                system = platform.system().lower()
                
                if system == 'darwin':  # macOS
                    possible_paths = [
                        '/usr/local/bin/wkhtmltopdf',
                        '/opt/homebrew/bin/wkhtmltopdf',
                        '/opt/local/bin/wkhtmltopdf'
                    ]
                elif system == 'linux':
                    possible_paths = [
                        '/usr/bin/wkhtmltopdf',
                        '/usr/local/bin/wkhtmltopdf'
                    ]
                elif system == 'windows':
                    possible_paths = [
                        r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe',
                        r'C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe'
                    ]
                else:
                    possible_paths = []
                
                # Check each possible path
                wkhtmltopdf_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        wkhtmltopdf_path = path
                        print(f"[INFO] Found wkhtmltopdf at: {wkhtmltopdf_path}")
                        break
                
                if wkhtmltopdf_path:
                    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
                else:
                    # Fall back to using without explicit path
                    print("[WARNING] wkhtmltopdf not found in common locations, trying default configuration")
                    config = None
        except Exception as e:
            print(f"[WARNING] Error configuring wkhtmltopdf path: {e}")
            config = None
        
        # Add CSS for print styling
        print_css = """
        <style>
            @page {
                size: letter;
                margin: 2cm;
            }
            body {
                font-family: Arial, sans-serif;
                line-height: 1.5;
                font-size: 11pt;
                color: #333;
            }
            h1 {
                font-size: 16pt;
                color: #204080;
                margin-top: 20pt;
                margin-bottom: 10pt;
                page-break-after: avoid;
            }
            h2 {
                font-size: 14pt;
                color: #204080;
                margin-top: 18pt;
                margin-bottom: 8pt;
                page-break-after: avoid;
            }
            h3 {
                font-size: 12pt;
                color: #204080;
                margin-top: 16pt;
                margin-bottom: 6pt;
                page-break-after: avoid;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 10pt 0;
                page-break-inside: avoid;
            }
            table, th, td {
                border: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
                text-align: left;
                padding: 6pt;
            }
            td {
                padding: 6pt;
            }
            .page-break {
                page-break-before: always;
            }
            img {
                max-width: 95%;
                height: auto;
                margin: 10pt auto;
                display: block;
            }
            p {
                text-align: justify;
                margin: 0 0 10pt 0;
            }
            code {
                font-family: Courier, monospace;
                background-color: #f5f5f5;
                padding: 2pt;
                border-radius: 3pt;
            }
            blockquote {
                margin: 10pt 20pt;
                padding: 10pt;
                background-color: #f9f9f9;
                border-left: 5pt solid #ccc;
            }
            li {
                margin-bottom: 5pt;
            }
            .summary-table, .trading-plan {
                page-break-inside: avoid;
            }
            .pdf-header {
                text-align: center;
                margin-bottom: 20pt;
                border-bottom: 1pt solid #ddd;
                padding-bottom: 10pt;
            }
            .pdf-footer {
                text-align: center;
                margin-top: 20pt;
                border-top: 1pt solid #ddd;
                padding-top: 10pt;
                font-size: 9pt;
                color: #666;
            }
            .watermark {
                position: fixed;
                bottom: 10pt;
                right: 10pt;
                opacity: 0.5;
                z-index: -1000;
                font-size: 8pt;
                color: #ccc;
            }
        </style>
        """
        
        # Add PDF header and footer
        header = f"""
        <div class="pdf-header">
            <h1>Advanced Stock Analysis Report</h1>
            <h2>{stock_symbol}</h2>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
        
        footer = """
        <div class="pdf-footer">
            <p>This report was generated using StockGPT's advanced analysis algorithms. The information provided is for informational purposes only and should not be considered financial advice.</p>
        </div>
        <div class="watermark">Generated by StockGPT</div>
        """
        
        # Combine all HTML parts
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Stock Analysis - {stock_symbol}</title>
            {print_css}
        </head>
        <body>
            {header}
            {html_content}
            {footer}
        </body>
        </html>
        """
        
        # Generate PDF - with or without explicit configuration
        try:
            if config:
                pdfkit.from_string(full_html, pdf_path, options=options, configuration=config)
            else:
                pdfkit.from_string(full_html, pdf_path, options=options)
            print(f"[INFO] Successfully generated PDF at {pdf_path}")
        except Exception as pdf_error:
            print(f"[ERROR] Failed to generate PDF with pdfkit: {pdf_error}")
            
            # Alternate approach - try to call wkhtmltopdf directly
            try:
                print("[INFO] Trying alternative PDF generation method...")
                html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                html_path = html_file.name
                
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(full_html)
                
                # Try to find wkhtmltopdf using subprocess
                try:
                    cmd = ['wkhtmltopdf', html_path, pdf_path]
                    subprocess.run(cmd, check=True)
                    print(f"[INFO] Successfully generated PDF using direct command at {pdf_path}")
                except Exception as cmd_error:
                    print(f"[ERROR] Failed to generate PDF using direct command: {cmd_error}")
                    raise
                finally:
                    os.unlink(html_path)  # Remove temporary HTML file
            except Exception as alt_error:
                print(f"[ERROR] Alternative PDF generation failed: {alt_error}")
                # Create a simple HTML file as a fallback
                with open(pdf_path + '.html', 'w', encoding='utf-8') as f:
                    f.write(full_html)
                print(f"[INFO] Created HTML file instead at {pdf_path}.html")
                return pdf_path + '.html'  # Return HTML path instead
        
        return pdf_path
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

@app.route('/download_pdf/<stock>', methods=['GET'])
def download_pdf(stock):
    """
    Generate and download analysis as PDF - reusing existing analysis data
    """
    try:
        # Get the existing analysis from the file cache instead of session
        from flask import session
        
        # Use reportlab for PDF generation - a pure Python solution that doesn't require external binaries
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from io import BytesIO
        import base64
        from html import unescape
        
        # Create a temporary file for the PDF
        pdf_buffer = BytesIO()
        
        # Setup the document
        doc = SimpleDocTemplate(
            pdf_buffer, 
            pagesize=letter,
            rightMargin=72, 
            leftMargin=72,
            topMargin=72, 
            bottomMargin=72
        )
        
        # Get the response data from cache
        gpt_response_html = None
        
        # Check if we're getting the correct stock
        session_stock = session.get('stock_symbol')
        
        # Check session for the cache filename
        cache_filename = session.get('response_cache_file')
        
        if cache_filename:
            print(f"[INFO] Found cache filename in session: {cache_filename}")
            gpt_response_html = load_response_from_cache(filename=cache_filename)
            
        # If not found via filename, try to find by stock symbol
        if not gpt_response_html:
            print(f"[INFO] Trying to load from cache by stock symbol: {stock}")
            gpt_response_html = load_response_from_cache(stock_symbol=stock)
            
        # If not found in cache, check g object
        if not gpt_response_html and hasattr(g, 'last_gpt_response') and hasattr(g, 'last_stock') and g.last_stock == stock:
            print(f"[INFO] Using GPT response from g object")
            gpt_response_html = g.last_gpt_response
        
        # If we still don't have it, return an error
        if not gpt_response_html:
            return "Error: No analysis found for this stock. Please run the analysis first and then download.", 400
        
        # Clean up HTML content - remove script tags and other potentially problematic elements
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(gpt_response_html, 'html.parser')
        
        # Remove script tags
        for script in soup.find_all('script'):
            script.extract()
            
        # Get cleaned HTML
        clean_html = str(soup)
        
        # Convert HTML to a simple text representation for PDF
        # This is a simplified approach - for complex HTML rendering consider using 
        # more advanced solutions like xhtml2pdf or pdfkit with a headless browser
        
        # Define styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        subtitle_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        ))
        
        styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=8
        ))
        
        # Create a list of flowables for the document
        flowables = []
        
        # Add title
        flowables.append(Paragraph(f"Stock Analysis Report - {stock.upper()}", styles['CustomHeading1']))
        flowables.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['CustomNormal']))
        flowables.append(Spacer(1, 0.25*inch))
        
        # Add summary paragraph
        summary = """This report contains detailed analysis of the stock, including technical indicators, 
        valuation metrics, price targets, and recommendations. The information provided is for educational 
        purposes only and should not be considered financial advice."""
        flowables.append(Paragraph(summary, styles['CustomNormal']))
        flowables.append(Spacer(1, 0.25*inch))
        
        # Extract and convert HTML content to reportlab flowables
        # This is a basic extraction that handles common HTML elements like headings and paragraphs
        
        for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'table']):
            if tag.name in ['h1', 'h2', 'h3']:
                text = tag.get_text().strip()
                if tag.name == 'h1':
                    flowables.append(Paragraph(text, styles['CustomHeading1']))
                elif tag.name == 'h2':
                    flowables.append(Paragraph(text, styles['CustomHeading2']))
                else:
                    flowables.append(Paragraph(text, styles['Heading3']))
                flowables.append(Spacer(1, 0.1*inch))
            
            elif tag.name == 'p':
                text = tag.get_text().strip()
                if text:
                    flowables.append(Paragraph(text, styles['CustomNormal']))
                    flowables.append(Spacer(1, 0.05*inch))
            
            elif tag.name in ['ul', 'ol']:
                for li in tag.find_all('li'):
                    text = li.get_text().strip()
                    # Add bullet point for unordered lists
                    if tag.name == 'ul':
                        text = f"• {text}"
                    # For ordered lists, we need to manually add numbers
                    flowables.append(Paragraph(text, styles['CustomNormal']))
                flowables.append(Spacer(1, 0.1*inch))
            
            elif tag.name == 'table':
                # Convert HTML table to ReportLab Table
                rows = []
                # First add header row
                header_row = []
                for th in tag.find_all('th'):
                    header_row.append(th.get_text().strip())
                
                if header_row:
                    rows.append(header_row)
                
                # Then add data rows
                for tr in tag.find_all('tr'):
                    row = []
                    for td in tr.find_all('td'):
                        row.append(td.get_text().strip())
                    if row:  # Only add if row is not empty
                        rows.append(row)
                
                if rows:
                    # Create the table
                    table = Table(rows)
                    
                    # Add table styling
                    style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ])
                    table.setStyle(style)
                    
                    flowables.append(table)
                    flowables.append(Spacer(1, 0.2*inch))
        
        # Add footer
        flowables.append(Spacer(1, 0.5*inch))
        flowables.append(Paragraph("StockGPT Analysis Report", styles['CustomNormal']))
        flowables.append(Paragraph("© " + str(datetime.now().year) + " StockGPT. All rights reserved.", styles['CustomNormal']))
        
        # Build the PDF document
        doc.build(flowables)
        
        # Get the PDF from the buffer
        pdf_data = pdf_buffer.getvalue()
        pdf_buffer.close()
        
        # Create response with PDF attachment
        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=StockGPT_Analysis_{stock}_{datetime.now().strftime("%Y%m%d")}.pdf'
        
        return response
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        
        # Return error as HTML if PDF generation fails
        error_html = f"""
        <html>
        <head><title>PDF Generation Error</title></head>
        <body>
            <h1>Error Generating PDF</h1>
            <p>There was an error generating your PDF report:</p>
            <pre>{str(e)}</pre>
            <p>Please try again or contact support if the issue persists.</p>
            <p><a href="/">Return to main page</a></p>
        </body>
        </html>
        """
        
        response = make_response(error_html)
        response.headers["Content-Type"] = "text/html"
        response.headers["Content-Disposition"] = f"inline; filename=error_report_{datetime.now().strftime('%Y%m%d')}.html"
        return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5003, debug=True)