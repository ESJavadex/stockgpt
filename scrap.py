from flask import Flask, render_template, request, send_file
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
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

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize OpenAI client
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

import markdown
import re

def chat_with_gpt(prompt):
    chat_completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    gpt_response = chat_completion.choices[0].message.content
    
    # Process the response to improve formatting
    processed_response = preprocess_markdown(gpt_response)
    
    # Convert markdown to HTML with extensions for tables and other formatting
    html_response = markdown.markdown(
        processed_response,
        extensions=[
            'markdown.extensions.tables',
            'markdown.extensions.fenced_code',
            'markdown.extensions.nl2br',
            'markdown.extensions.sane_lists'
        ]
    )
    
    # Further enhance the HTML with additional styling
    html_response = postprocess_html(html_response)
    
    return html_response

def preprocess_markdown(text):
    """Preprocess markdown to improve formatting before conversion"""
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
        
        # Fix summary table formatting
        if 'Entry Price' in line and 'Direction' in line and 'TP' in line and 'SL' in line:
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
    text = re.sub(r'(\n[0-9]+\.)([^\n])', r'\1 \2', text)  # Add space after numbered list markers
    text = re.sub(r'(\n- )([^\n])', r'\1\2', text)  # Ensure proper formatting for bullet lists
    
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
    
    return '\n'.join(lines)

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
    
    # Replace newlines with spaces instead of <br> tags in HTML content outside pre/code blocks
    chunks = re.split(r'(<pre>.*?</pre>|<code>.*?</code>)', html, flags=re.DOTALL)
    for i in range(0, len(chunks), 2):  # Process only non-code/pre chunks
        chunks[i] = chunks[i].replace('\n', ' ')
    html = ''.join(chunks)
    
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
    
    # Enhance headings
    html = re.sub(r'<h1>(.*?)</h1>', r'<h1 class="text-primary">\1</h1>', html)
    html = re.sub(r'<h2>(.*?)</h2>', r'<h2 class="text-secondary">\1</h2>', html)
    html = re.sub(r'<h3>(.*?)</h3>', r'<h3 class="text-info">\1</h3>', html)
    
    # Style the Executive Summary section
    html = re.sub(r'<h[1-6]\s+class="[^"]*">\s*Executive Summary\s*</h[1-6]>', 
                 r'<div class="card mb-4"><div class="card-header bg-primary text-white">Executive Summary</div><div class="card-body">', 
                 html, flags=re.IGNORECASE)
    if 'Executive Summary' in html:
        # Find the next heading to close the card
        next_heading = re.search(r'<h[1-6]', html[html.find('Executive Summary') + 20:])
        if next_heading:
            pos = html.find('Executive Summary') + 20 + next_heading.start()
            html = html[:pos] + '</div></div>' + html[pos:]
        else:
            html += '</div></div>'
    
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
    stock = yf.Ticker(stock_symbol)
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

@app.route('/', methods=['GET', 'POST'])
def index_scrap():
    if request.method == 'POST':
        stock = request.form['stock']
        routes = ["cheat-sheet", "technical-analysis", "performance", "analyst-ratings"]
        #routes = ["cheat-sheet"]
        scraped_content = {}

        driver = webdriver.Chrome()

        try:
            for route in routes:
                url = f"https://www.barchart.com/stocks/quotes/{stock}/{route}"
                driver.get(url)
                wait = WebDriverWait(driver, 10)
                main_content_div = wait.until(EC.presence_of_element_located((By.ID, "main-content-column")))
                scraped_content[route] = main_content_div.text.strip()

            stock_data = get_stock_data(stock)
            
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
            {' '.join(scraped_content.values())}

            Based on this comprehensive data, please provide:

            0. SUMMARTY OF THE STOCK MARKET AND INDUSTRY
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

            3. FUNDAMENTAL OUTLOOK:
               - Assessment of financial health and business model strength
               - Growth prospects and competitive positioning
               - Earnings quality and sustainability analysis
               - Risk factors that could impact future performance
               - Dividend sustainability and capital return outlook (if applicable)

            4. INVESTMENT STRATEGY:
               - Clear buy/sell/hold recommendation with detailed rationale
               - Multiple entry strategies with specific price points
               - Multiple exit strategies including profit targets and time horizons
               - Risk management approach with specific stop-loss levels
               - Position sizing recommendations based on risk/reward profile

            5. TRADING STRATEGY:
               - Clear buy/sell/hold recommendation with detailed rationale
               - Multiple entry strategies with specific price points
               - Multiple exit strategies including profit targets and time horizons
               - Risk management approach with specific stop-loss levels
               - Position sizing recommendations based on risk/reward profile

            6. CONSIDERATIONS:
               - Market conditions and potential catalysts
               - Risk factors and mitigation strategies
               - Liquidity and market access
               - Transaction costs and fees
               - Time horizon and market volatility
               
            7. EXECUTIVE SUMMARY:
               Entry price: [specific price or price range]
               Trade direction: [Buy/Sell/Hold]
               Take Profit (TP): [multiple levels with percentages]
               Stop Loss (SL): [specific level with percentage]
               Risk/Reward Ratio: [calculated ratio]
               Time Horizon: [short-term/medium-term/long-term]
               Conviction Level: [high/medium/low]

            8. FINAL CONCLUSION: 
               - Summary of the analysis
               - Key takeaways
               - Next steps
               - Additional recommendations

            9. OTHER:
               - Any additional information or insights
            
            10. Summary of the analysis in one line

            Please be specific with price levels and percentages throughout your analysis. Consider transaction costs, market volatility, and liquidity in your recommendations.
            
            At the end write a FULL Executive Summary of the analysis (in spanish)

            Ensure that:
            - The analysis is based on the data provided and you have to do it like the investor never bought the stock.
            - All sections are detailed and data-driven  
            - Prices, targets, and strategies are **quantified**  
            - Writing is **professional and clear**
            - Everything is explained in detail
            - Write at least 5000 words of narrative, with clear sub‑headings under each section.
            - Explain everything in detail, deep analysis.
            - Do the report in spanish. Keep in english the words you want. 
            
            """
            print("PROMPT:", prompt)
            gpt_response_html= chat_with_gpt(prompt)
            print(gpt_response_html)

            return render_template('index_scrap.html', now=datetime.now, stock=stock, scraped_content=scraped_content, gpt_response=gpt_response_html, stock_data=stock_data)

        
        finally:
            driver.quit()
    
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

if __name__ == '__main__':
    app.run(port=5007, debug=True)