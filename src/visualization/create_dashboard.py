#!/usr/bin/env python3
"""
Dashboard creation script for Oil Market Forecasting PoC.
This script creates interactive dashboards using Plotly and Dash
to visualize oil market forecasts and historical data.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OilMarketDashboard:
    """Class to create and manage oil market dashboards."""
    
    def __init__(self):
        """Initialize dashboard components."""
        self.app = Dash(__name__)
        self.historical_data = None
        self.forecast_data = None
        
    def load_data(self, historical_file: str, forecast_file: str):
        """Load historical and forecast data."""
        try:
            # Load historical data
            with open(historical_file, 'r') as f:
                historical_data = json.load(f)
            self.historical_data = pd.DataFrame(historical_data)
            
            # Load forecast data
            with open(forecast_file, 'r') as f:
                forecast_data = json.load(f)
            self.forecast_data = pd.DataFrame(forecast_data)
            
            # Convert dates
            self.historical_data['date'] = pd.to_datetime(self.historical_data['date'])
            self.forecast_data['date'] = pd.to_datetime(self.forecast_data['date'])
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def create_price_trend_figure(self) -> go.Figure:
        """Create price trend visualization."""
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(go.Scatter(
            x=self.historical_data['date'],
            y=self.historical_data['price'],
            name='Historical Price',
            line=dict(color='blue')
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=self.forecast_data['date'],
            y=self.forecast_data['predicted_price'],
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Add confidence interval
        upper_bound = self.forecast_data['predicted_price'] * (1 + self.forecast_data['confidence_score'])
        lower_bound = self.forecast_data['predicted_price'] * (1 - self.forecast_data['confidence_score'])
        
        fig.add_trace(go.Scatter(
            x=self.forecast_data['date'],
            y=upper_bound,
            fill=None,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=self.forecast_data['date'],
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line=dict(width=0),
            name='Confidence Interval',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig.update_layout(
            title='Oil Price Trend and Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified'
        )
        
        return fig
        
    def create_price_distribution_figure(self) -> go.Figure:
        """Create price distribution visualization."""
        fig = go.Figure()
        
        # Historical price distribution
        fig.add_trace(go.Histogram(
            x=self.historical_data['price'],
            name='Historical Prices',
            opacity=0.7,
            nbinsx=30
        ))
        
        # Forecast price distribution
        fig.add_trace(go.Histogram(
            x=self.forecast_data['predicted_price'],
            name='Forecast Prices',
            opacity=0.7,
            nbinsx=30
        ))
        
        fig.update_layout(
            title='Price Distribution',
            xaxis_title='Price (USD)',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        return fig
        
    def create_confidence_trend_figure(self) -> go.Figure:
        """Create confidence score trend visualization."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.forecast_data['date'],
            y=self.forecast_data['confidence_score'],
            mode='lines+markers',
            name='Confidence Score'
        ))
        
        fig.update_layout(
            title='Forecast Confidence Trend',
            xaxis_title='Date',
            yaxis_title='Confidence Score',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
        
    def setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1('Oil Market Forecast Dashboard',
                   style={'textAlign': 'center', 'marginBottom': '30px'}),
            
            # Date range selector
            html.Div([
                html.Label('Select Date Range:'),
                dcc.DatePickerRange(
                    id='date-range',
                    min_date_allowed=self.historical_data['date'].min(),
                    max_date_allowed=self.forecast_data['date'].max(),
                    start_date=self.historical_data['date'].max() - timedelta(days=30),
                    end_date=self.forecast_data['date'].max()
                )
            ], style={'marginBottom': '20px'}),
            
            # Price trend graph
            html.Div([
                dcc.Graph(
                    id='price-trend-graph',
                    figure=self.create_price_trend_figure()
                )
            ], style={'marginBottom': '30px'}),
            
            # Price distribution and confidence trend
            html.Div([
                html.Div([
                    dcc.Graph(
                        id='price-distribution-graph',
                        figure=self.create_price_distribution_figure()
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        id='confidence-trend-graph',
                        figure=self.create_confidence_trend_figure()
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Summary statistics
            html.Div([
                html.H3('Summary Statistics'),
                html.Div(id='summary-stats')
            ], style={'marginTop': '30px'})
        ])
        
    def update_summary_stats(self, start_date: datetime, end_date: datetime) -> html.Div:
        """Update summary statistics based on selected date range."""
        # Filter data for selected range
        historical_mask = (self.historical_data['date'] >= start_date) & \
                         (self.historical_data['date'] <= end_date)
        forecast_mask = (self.forecast_data['date'] >= start_date) & \
                       (self.forecast_data['date'] <= end_date)
                       
        hist_data = self.historical_data[historical_mask]
        fore_data = self.forecast_data[forecast_mask]
        
        # Calculate statistics
        stats = {
            'Average Historical Price': f"${hist_data['price'].mean():.2f}",
            'Average Forecast Price': f"${fore_data['predicted_price'].mean():.2f}",
            'Price Range': f"${hist_data['price'].min():.2f} - ${hist_data['price'].max():.2f}",
            'Average Confidence Score': f"{fore_data['confidence_score'].mean():.1%}"
        }
        
        # Create summary div
        return html.Div([
            html.Div([
                html.Strong(f"{key}: "),
                html.Span(f"{value}")
            ]) for key, value in stats.items()
        ])
        
    def setup_callbacks(self):
        """Set up interactive callbacks."""
        @self.app.callback(
            [Output('price-trend-graph', 'figure'),
             Output('price-distribution-graph', 'figure'),
             Output('confidence-trend-graph', 'figure'),
             Output('summary-stats', 'children')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_graphs(start_date, end_date):
            # Convert string dates to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            
            # Update figures
            trend_fig = self.create_price_trend_figure()
            dist_fig = self.create_price_distribution_figure()
            conf_fig = self.create_confidence_trend_figure()
            
            # Update date ranges for all figures
            for fig in [trend_fig, dist_fig, conf_fig]:
                fig.update_layout(
                    xaxis_range=[start_date, end_date]
                )
            
            # Update summary stats
            summary_stats = self.update_summary_stats(start_date, end_date)
            
            return trend_fig, dist_fig, conf_fig, summary_stats
            
    def run_server(self, debug: bool = False, port: int = 8050):
        """Run the dashboard server."""
        self.app.run_server(debug=debug, port=port)

def main():
    """Main function to create and run dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Oil Market Forecast Dashboard')
    parser.add_argument('--historical-data', required=True,
                      help='Path to historical data JSON file')
    parser.add_argument('--forecast-data', required=True,
                      help='Path to forecast data JSON file')
    parser.add_argument('--port', type=int, default=8050,
                      help='Port to run the dashboard server')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode')
    args = parser.parse_args()
    
    try:
        # Create dashboard
        dashboard = OilMarketDashboard()
        
        # Load data
        dashboard.load_data(args.historical_data, args.forecast_data)
        
        # Setup dashboard
        dashboard.setup_layout()
        dashboard.setup_callbacks()
        
        # Run server
        logger.info(f"Starting dashboard server on port {args.port}")
        dashboard.run_server(debug=args.debug, port=args.port)
        
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        raise

if __name__ == '__main__':
    main() 