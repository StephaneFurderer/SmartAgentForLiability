# 🚀 Streamlit Cloud Deployment Guide

## Prerequisites
- GitHub repository with your code
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))

## Deployment Steps

### 1. Repository Setup
- Ensure your main app file is named `notes_reviewer_app.py`
- All dependencies are listed in `requirements.txt`
- Data files are included in the `_data/` directory

### 2. Streamlit Cloud Setup
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `StephaneFurderer/SmartAgentForLiability`
5. Set the main file path: `notes_reviewer_app.py`
6. Set the app URL (optional)
7. Click "Deploy!"

### 3. Environment Variables
Set these in Streamlit Cloud → Settings → Secrets:
```toml
OPENAI_API_KEY = "your_openai_api_key_here"
```

### 4. App Configuration
The app will automatically use the `.streamlit/config.toml` file for:
- Theme customization
- Server settings
- Browser configuration

## Troubleshooting

### Common Issues:
1. **Import Errors**: Ensure all packages are in `requirements.txt`
2. **Data Not Found**: Check that `_data/` directory is included
3. **API Key Issues**: Verify environment variables are set correctly

### Dependencies Included:
- `pandas>=2.0.0` - Data manipulation
- `streamlit>=1.28.0` - Web framework
- `numpy>=1.24.0` - Numerical computing
- `openai>=1.0.0` - OpenAI API client
- `plotly>=5.0.0` - Interactive charts
- `pyarrow>=10.0.0` - Parquet file support

## App Features
✅ **Interactive Claims Table** - Click to select claims  
✅ **AI-Powered Summary** - Professional claim analysis  
✅ **Communication Timeline** - Filterable notes and emails  
✅ **Financial Visualization** - Claim lifetime charts  
✅ **Chat Interface** - Ask questions about claims  

## Support
If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all dependencies are installed
3. Ensure data files are accessible
4. Test locally first with `streamlit run notes_reviewer_app.py`
