# OT Analytics Finance System - Complete Setup Guide

This guide provides step-by-step installation and configuration instructions for the complete OT Analytics Finance System.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Detailed Installation](#detailed-installation)
4. [Component Setup](#component-setup)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (for parallel data fetching)

### Software
- **Python**: 3.11 or higher
- **OS**: Linux, macOS, or Windows
- **Optional**: Docker (for containerized deployment)

### External Services
- **Alpha Vantage API Key** (free) - For news and enhanced data
- **Ollama** (optional) - For local AI model inference
- **Discord Bot Token** (optional) - For Discord integration

---

## Quick Start (5 Minutes)

For experienced users who want to get started immediately:

```bash
# 1. Clone repository
cd /home/user/finance

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env and add your ALPHA_VANTAGE_KEY

# 4. Run the OT Analytics Terminal
python ot_analytics_terminal.py

# 5. (Optional) Install Ollama for AI features
curl https://ollama.ai/install.sh | sh
ollama pull llama3:8b
ollama create ot-analytics-quant -f Modelfile
```

---

## Detailed Installation

### Step 1: Python Environment Setup

#### Option A: Virtual Environment (Recommended)

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

#### Option B: Conda Environment

```bash
# Create conda environment
conda create -n ot-analytics python=3.11
conda activate ot-analytics
```

### Step 2: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Dependencies installed:**
- `yfinance>=0.2.40` - Real-time stock data
- `pandas>=2.2.0`, `numpy>=1.26` - Data manipulation
- `scipy>=1.11`, `statsmodels>=0.14` - Statistical analysis
- `arch>=6.3` - GARCH models
- `scikit-learn>=1.4` - Machine learning
- `matplotlib>=3.8`, `seaborn>=0.13` - Visualization
- `chromadb>=0.4.22` - Vector database for RAG
- `pypdf>=3.17.0` - PDF processing
- `langchain-text-splitters>=0.0.1` - Text chunking
- `discord.py>=2.3` - Discord bot (optional)
- `pydantic>=2.6` - Data validation
- `python-dotenv>=1.0` - Environment variables

### Step 3: Get Alpha Vantage API Key

1. Visit: https://www.alphavantage.co/support/#api-key
2. Click "GET YOUR FREE API KEY TODAY"
3. Fill in basic information (email, use case)
4. Copy your API key

**Free Tier Limits:**
- 5 API calls per minute
- 500 API calls per day
- Sufficient for testing and light usage

### Step 4: Create Environment File

```bash
# Copy example file
cp .env.example .env

# Edit .env file
nano .env  # or vim, or your preferred editor
```

**Required variables:**
```bash
# Alpha Vantage (for news and enhanced data)
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Optional: For AI agent features
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: For Discord bot
DISCORD_TOKEN=your_discord_bot_token_here
DISCORD_CHANNELS=channel_id_1,channel_id_2
```

### Step 5: Install Ollama (Optional but Recommended)

Ollama provides local AI inference for the OT-Analytics-Quant-Model.

#### Linux

```bash
curl https://ollama.ai/install.sh | sh
```

#### macOS

```bash
# Download from https://ollama.ai
# Or use Homebrew:
brew install ollama
```

#### Windows

Download installer from: https://ollama.ai/download/windows

#### Create Custom Model

```bash
# Pull base model
ollama pull llama3:8b

# Create custom quant model
cd /home/user/finance
ollama create ot-analytics-quant -f Modelfile

# Verify installation
ollama list
# Should show: ot-analytics-quant
```

---

## Component Setup

The system consists of three main components (wrappers):

### 1. Stock Training Database (Wrapper 1)

Historical stock analysis with regime detection.

```bash
# Initial training run (5-10 minutes)
python stock_training_db.py --train

# Access interactive menu
python stock_training_db.py --chat
```

**Features:**
- Menu options 1-70
- Historical analysis of 20 tech stocks
- Regime detection and risk metrics
- PDF ingestion (option 71)

### 2. Unified Terminal (Wrapper 2)

Real-time stock data with AI analysis.

```bash
# Interactive mode
python unified_terminal.py

# Demo mode
python unified_terminal.py --demo
```

**Features:**
- Real-time data for Tech, AI, All categories
- Custom stock lists
- AI analysis
- Export to CSV

### 3. OT Analytics Terminal (Wrapper 3) ⭐ **MAIN**

Complete system with auto-loading, news, and RAG.

```bash
# Interactive mode
python ot_analytics_terminal.py

# Demo with automatic briefing
python ot_analytics_terminal.py --demo
```

**Features:**
- Everything from Wrappers 1 & 2
- Automatic data loading
- News sentiment analysis
- Ollama AI integration
- RAG memory (PDF Q&A)
- Comprehensive context for queries

### 4. Discord Bot (Optional)

Live market scanner posting to Discord.

```bash
# Run Discord bot
python main.py --channels YOUR_CHANNEL_ID

# Or use environment variables
python main.py
```

---

## Configuration

### Database Locations

The system creates several databases:

```
/home/user/finance/
├── stock_training.db      # Training database (Wrapper 1)
├── pdf_metadata.db        # PDF metadata (SQLite)
├── chroma_db/             # Vector embeddings (ChromaDB)
│   ├── chroma.sqlite3
│   └── [embedding files]
```

### Model Parameters

Edit `Modelfile` to adjust AI model parameters:

```modelfile
# Ultra-precise for quant work
PARAMETER temperature 0.05
PARAMETER top_p 0.85
PARAMETER repeat_penalty 1.15
PARAMETER presence_penalty 0.0
PARAMETER frequency_penalty 0.0
```

### Customizing Stock Lists

Edit terminal files to add custom stock categories:

```python
# In unified_terminal.py or ot_analytics_terminal.py
CUSTOM_STOCKS = [
    "YOUR", "CUSTOM", "TICKERS", "HERE"
]
```

---

## Verification

### Test Installation

```bash
# Test Python dependencies
python -c "import yfinance, pandas, chromadb; print('✅ Dependencies OK')"

# Test Ollama (if installed)
ollama list | grep ot-analytics-quant && echo "✅ Ollama OK"

# Test database creation
python stock_training_db.py --train
# Should complete without errors

# Test PDF system
python test_pdf_simple.py
# Should show: "🎉 All tests passed!"
```

### Test Each Component

#### Test Training Database

```bash
python stock_training_db.py --chat
# In the menu, try option 1 (Top Performers)
```

#### Test Unified Terminal

```bash
python unified_terminal.py
# At prompt: show tech
# Should display 20 tech stocks
```

#### Test OT Analytics Terminal

```bash
python ot_analytics_terminal.py
# Should auto-load data and show status
# At prompt: auto
# Should generate market briefing
```

### Verify Data Loading

After running OT Analytics Terminal, you should see:

```
✅ Ollama connected: ot-analytics-quant
✅ Training database: 20 stocks (Run ID: 1)
✅ RAG Memory: Ready
✅ Loaded 20 news articles
✅ Market snapshot: 5 securities
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'chromadb'
```

**Solution:**
```bash
pip install chromadb pypdf langchain-text-splitters
```

#### 2. Ollama Not Available

**Problem:**
```
⚠️  Ollama not available
```

**Solution:**
```bash
# Check if Ollama is running
ollama list

# If not, start Ollama service
systemctl start ollama  # Linux
# or restart Ollama app (macOS/Windows)

# Recreate model
ollama create ot-analytics-quant -f Modelfile
```

#### 3. News Not Loading

**Problem:**
```
⚠️  News unavailable
```

**Solution:**
```bash
# Check API key in .env
cat .env | grep ALPHA_VANTAGE_KEY

# If missing, add it:
echo "ALPHA_VANTAGE_KEY=YOUR_KEY" >> .env

# Restart terminal
```

#### 4. yfinance Issues

**Problem:**
```
Error importing PDFVectorStore: No module named 'yfinance'
```

**Solution:**
```bash
# Install yfinance without problematic dependencies
pip install yfinance --no-deps
pip install requests lxml beautifulsoup4 html5lib

# Or try:
pip install --upgrade yfinance
```

#### 5. ChromaDB Errors

**Problem:**
```
Error: Could not connect to ChromaDB
```

**Solution:**
```bash
# Reinstall ChromaDB
pip uninstall chromadb
pip install chromadb>=0.4.22

# Check permissions
chmod -R 755 chroma_db/
```

#### 6. Training Database Empty

**Problem:**
```
⚠️  No training data found!
```

**Solution:**
```bash
# Run training (takes 5-10 minutes)
python stock_training_db.py --train

# This will:
# - Fetch historical data for 20 stocks
# - Compute technical indicators
# - Detect regime changes
# - Store in SQLite database
```

### Performance Issues

#### Slow Startup

- **Cause**: News API or stock data fetching
- **Solution**: Check internet connection, or disable news temporarily

#### Slow AI Responses

- **Cause**: Large context or slow model
- **Solution**:
  - Use smaller context (limit news items)
  - Upgrade hardware (more RAM for Ollama)
  - Use streaming mode for better UX

#### High Memory Usage

- **Cause**: Ollama model loaded in RAM
- **Solution**:
  - Normal: 4-8GB for llama3:8b
  - Close other applications
  - Consider smaller model if needed

### Getting Help

1. **Check logs**: Most errors print detailed tracebacks
2. **Verify environment**: `pip list | grep chromadb`
3. **Test components separately**: Use test scripts
4. **Check GitHub issues**: For known bugs
5. **Review documentation**: Each component has detailed README

---

## Next Steps

After successful installation:

1. **Run initial training**: `python stock_training_db.py --train`
2. **Ingest PDFs**: Use option 71 to add research documents
3. **Explore features**: Try `auto` command in OT Analytics Terminal
4. **Customize**: Adjust stock lists, model parameters
5. **Automate**: Set up cron jobs for daily training

---

## Security Best Practices

### API Keys

- **Never commit** `.env` file to version control
- **Rotate keys** periodically
- **Use environment variables** in production
- **Limit permissions** on config files

```bash
# Secure .env file
chmod 600 .env
```

### Database Security

- **Backup regularly**: Databases contain valuable analysis
- **Encrypt if needed**: Use disk encryption for sensitive data
- **Limit access**: Only authorized users

```bash
# Backup databases
cp stock_training.db backups/stock_training_$(date +%Y%m%d).db
tar -czf chroma_backup.tar.gz chroma_db/
```

### Network Security

- **Use HTTPS**: For all API calls
- **Firewall rules**: If exposing services
- **VPN**: For remote access

---

## Production Deployment

### Systemd Service (Linux)

Create `/etc/systemd/system/ot-analytics.service`:

```ini
[Unit]
Description=OT Analytics Terminal
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/home/user/finance
Environment="PATH=/home/user/finance/venv/bin"
ExecStart=/home/user/finance/venv/bin/python ot_analytics_terminal.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ot-analytics
sudo systemctl start ot-analytics
```

### Docker (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "ot_analytics_terminal.py"]
```

Build and run:
```bash
docker build -t ot-analytics .
docker run -it --env-file .env ot-analytics
```

### Cron Jobs

Daily training at 8 AM:

```bash
# Edit crontab
crontab -e

# Add line:
0 8 * * * cd /home/user/finance && /home/user/finance/venv/bin/python stock_training_db.py --train
```

---

## Performance Tuning

### Optimize Data Fetching

```python
# In stock_training_db.py or custom scripts
# Increase parallel fetching
import concurrent.futures

with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(fetch_stock, ticker) for ticker in tickers]
```

### Reduce Memory Usage

```bash
# Use smaller Ollama model
ollama pull llama3:8b-q4_0  # Quantized version
```

### Cache Data

```python
# Implement caching for API calls
import requests_cache
requests_cache.install_cache('alpha_vantage_cache', expire_after=3600)
```

---

## Maintenance

### Regular Tasks

**Daily:**
- Run training: `python stock_training_db.py --train`
- Check logs for errors
- Verify API quota usage

**Weekly:**
- Update stock lists
- Review and clean old data
- Backup databases

**Monthly:**
- Update dependencies: `pip install -r requirements.txt --upgrade`
- Rotate API keys if needed
- Review and optimize performance

### Updates

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Recreate Ollama model if Modelfile changed
ollama create ot-analytics-quant -f Modelfile
```

---

## Support Resources

### Documentation
- `readme.md` - Project overview
- `OT_ANALYTICS_README.md` - Terminal guide
- `PDF_VECTOR_DB_SCHEMA.md` - RAG system docs
- `UNIFIED_TERMINAL_README.md` - Wrapper 2 guide
- `STOCK_TRAINING_DB_README.md` - Wrapper 1 guide

### Test Scripts
- `test_pdf_simple.py` - Test PDF system
- `test_pdf_system.py` - Comprehensive PDF tests
- `test_unified_terminal.py` - Terminal tests

### Quick Reference
See OT_ANALYTICS_README.md for command reference card

---

## Appendix: Component Comparison

| Feature | Wrapper 1 | Wrapper 2 | Wrapper 3 |
|---------|-----------|-----------|-----------|
| **File** | stock_training_db.py | unified_terminal.py | ot_analytics_terminal.py |
| Training Database | ✅ | ✅ | ✅ |
| Menu Options 1-70 | ✅ | ✅ | ✅ |
| PDF Ingestion (71) | ✅ | ✅ | ✅ |
| Real-time Data | ❌ | ✅ | ✅ |
| News Sentiment | ❌ | ❌ | ✅ |
| Ollama AI | ❌ | ❌ | ✅ |
| RAG Memory | ❌ | ❌ | ✅ |
| Auto-Loading | ❌ | ❌ | ✅ |
| **Recommended For** | Historical analysis | Live trading | Production use |

---

**Setup complete! You're ready to use the OT Analytics Finance System.**

For questions or issues, refer to the troubleshooting section or review component-specific documentation.
