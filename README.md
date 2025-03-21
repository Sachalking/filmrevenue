# Film Revenue Predictor

A Flask web application that predicts worldwide box office revenue based on a film's production budget using machine learning.

## Features

- Linear regression model trained on historical movie budget and revenue data
- Clean, responsive UI with modern design
- Simple input form for budget prediction
- Clear results display

## Deployment to Vercel

This application is configured for deployment on Vercel. Follow these steps to deploy:

### Prerequisites

1. [GitHub Account](https://github.com/)
2. [Vercel Account](https://vercel.com/) (you can sign up with your GitHub account)
3. [Vercel CLI](https://vercel.com/cli) (optional for local development)

### Deployment Steps

1. **Push your code to GitHub**
   
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/your-username/film-revenue-predictor.git
   git push -u origin main
   ```

2. **Deploy with Vercel**

   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect it's a Flask application
   - Click "Deploy"

### Local Development

To run the application locally:

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask application:
   ```bash
   flask run
   ```

3. Open your browser and navigate to `http://127.0.0.1:5000`

## Data Source

The application uses the `cost-revenue-clean.csv` dataset which contains historical movie budget and revenue data. #   f i l m r e v e n u e  
 