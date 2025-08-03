# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your MLOps Secure Churn Prediction app to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account**: Your code must be in a GitHub repository
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Python Dependencies**: All required packages are listed in `requirements.txt`

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

Ensure your repository has the following files:
- ✅ `streamlit_app.py` (main Streamlit application)
- ✅ `requirements.txt` (Python dependencies)
- ✅ `.streamlit/config.toml` (Streamlit configuration)

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**: Visit [share.streamlit.io](https://share.streamlit.io)

2. **Sign in with GitHub**: Connect your GitHub account

3. **New App**: Click "New app"

4. **Configure Deployment**:
   - **Repository**: Select your GitHub repository
   - **Branch**: Choose `main` (or your default branch)
   - **Main file path**: Enter `streamlit_app.py`
   - **App URL**: Choose a unique URL for your app

5. **Deploy**: Click "Deploy!"

### Step 3: Monitor Deployment

- **Build Status**: Watch the build logs for any errors
- **Runtime**: The app will be available at your chosen URL
- **Updates**: Any push to your main branch will automatically redeploy

## 🔧 Configuration Options

### Environment Variables (Optional)

You can add environment variables in Streamlit Cloud:
- Go to your app settings
- Add any required API keys or configuration

### Custom Domain (Optional)

- Contact Streamlit support for custom domain setup
- Requires business plan

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure all packages are in `requirements.txt`
   - Check for version conflicts

2. **Memory Issues**:
   - Reduce data size in `generate_sample_data()`
   - Use `@st.cache_data` for expensive operations

3. **Timeout Errors**:
   - Optimize model training with smaller datasets
   - Use caching for expensive computations

### Performance Tips

1. **Caching**: Use `@st.cache_data` and `@st.cache_resource`
2. **Data Size**: Limit sample data size for faster loading
3. **Lazy Loading**: Load heavy components only when needed

## 📊 App Features

Your deployed app includes:

- **🏠 Overview**: Project introduction and key metrics
- **📊 Data Exploration**: Interactive data visualization
- **🤖 Model Predictions**: Real-time churn predictions
- **🔍 Model Explainability**: SHAP analysis and feature importance
- **📈 Business Insights**: Actionable recommendations and ROI analysis

## 🔐 Security Notes

- **No Sensitive Data**: The app uses synthetic data only
- **PII Protection**: All customer data is anonymized
- **Compliance**: Follows GDPR and CCPA guidelines

## 📈 Monitoring

- **Usage Analytics**: Available in Streamlit Cloud dashboard
- **Performance**: Monitor app response times
- **Errors**: Check logs for any issues

## 🔄 Updates

To update your app:
1. Make changes to your code
2. Push to GitHub
3. Streamlit Cloud automatically redeploys

## 📞 Support

- **Streamlit Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repository

---

**🎉 Congratulations!** Your MLOps Secure Churn Prediction app is now live on Streamlit Cloud! 