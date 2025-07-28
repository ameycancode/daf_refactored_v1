import plotly.graph_objs as go
import pandas as pd
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def visualize_results_container(processed_test_sets, output_path, current_date, show_individual=False):
    """
    Visualizes predictions and saves results for SageMaker container.
    Modified version of the original visualize_results function.
    """
    try:
        if not processed_test_sets:
            logger.warning("No data available to visualize.")
            return None
        
        logger.info(f"Visualizing results for {len(processed_test_sets)} profiles")
        
        # Create individual plots if requested
        if show_individual:
            _create_individual_plots(processed_test_sets, output_path, current_date)
        
        # Create combined plot
        _create_combined_plot(processed_test_sets, output_path, current_date)
        
        # Create aggregated results
        combined_df, aggregated_df = _create_aggregated_data(processed_test_sets)
        
        # Save combined and aggregated results
        _save_results_to_files(combined_df, aggregated_df, output_path, current_date)
        
        # Create aggregated plot
        _create_aggregated_plot(aggregated_df, output_path, current_date)
        
        logger.info("Visualization and results saving completed successfully")
        
        return {
            'combined_data': combined_df,
            'aggregated_data': aggregated_df,
            'profile_count': len(processed_test_sets)
        }
        
    except Exception as e:
        logger.error(f"Visualization failed: {str(e)}")
        return None

def _create_individual_plots(processed_test_sets, output_path, current_date):
    """Create individual plots for each profile"""
    try:
        plots_dir = os.path.join(output_path, 'individual_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        for dataset_name, df_test in processed_test_sets.items():
            fig = go.Figure(go.Scatter(
                x=df_test['TradeDateTime'],
                y=df_test['Load_All'],
                mode='lines',
                name=f"{dataset_name} - Predicted Load",
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f"Predicted Load for {dataset_name}",
                xaxis_title="Time",
                yaxis_title="Load (kWh)",
                template="plotly_white",
                hovermode="x unified",
                width=1000,
                height=500
            )
            
            # Save plot as HTML
            plot_file = os.path.join(plots_dir, f"{dataset_name}_prediction_{current_date}.html")
            fig.write_html(plot_file)
            
            logger.info(f"Individual plot saved for {dataset_name}")
            
    except Exception as e:
        logger.error(f"Individual plots creation failed: {str(e)}")

def _create_combined_plot(processed_test_sets, output_path, current_date):
    """Create combined plot for all profiles"""
    try:
        combined_fig = go.Figure()
        
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ]
        
        for i, (dataset_name, df_test) in enumerate(processed_test_sets.items()):
            color = colors[i % len(colors)]
            
            combined_fig.add_trace(go.Scatter(
                x=df_test['TradeDateTime'],
                y=df_test['Load_All'],
                mode='lines',
                name=dataset_name,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{dataset_name}</b><br>" +
                             "Time: %{x}<br>" +
                             "Load: %{y:,.2f} kWh<br>" +
                             "<extra></extra>"
            ))
        
        combined_fig.update_layout(
            title=f"Combined Predicted Load for All Profiles - {current_date}",
            xaxis_title="Time",
            yaxis_title="Load (kWh)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            width=1200,
            height=600
        )
        
        # Save combined plot
        combined_plot_file = os.path.join(output_path, f"combined_predictions_{current_date}.html")
        combined_fig.write_html(combined_plot_file)
        
        logger.info("Combined plot created and saved")
        
    except Exception as e:
        logger.error(f"Combined plot creation failed: {str(e)}")

def _create_aggregated_data(processed_test_sets):
    """Create combined and aggregated data"""
    try:
        # Combine all profiles into one DataFrame
        combined_df = pd.concat([
            df.assign(Profile=name) for name, df in processed_test_sets.items()
        ], ignore_index=True)
        
        # Create aggregated data (sum of Load_All across profiles)
        aggregated_df = combined_df.groupby('TradeDateTime', as_index=False).agg({
            'Load_All': 'sum',
            'Predicted_Load': 'sum',
            'Count': 'sum'
        })
        
        # Add time components
        aggregated_df['Hour'] = aggregated_df['TradeDateTime'].dt.hour
        aggregated_df['Date'] = aggregated_df['TradeDateTime'].dt.date
        
        logger.info(f"Created combined data: {len(combined_df)} records")
        logger.info(f"Created aggregated data: {len(aggregated_df)} records")
        
        return combined_df, aggregated_df
        
    except Exception as e:
        logger.error(f"Data aggregation failed: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

def _create_aggregated_plot(aggregated_df, output_path, current_date):
    """Create aggregated plot"""
    try:
        if aggregated_df.empty:
            logger.warning("No aggregated data to plot")
            return
        
        aggregated_fig = go.Figure(go.Scatter(
            x=aggregated_df['TradeDateTime'],
            y=aggregated_df['Load_All'],
            mode='lines+markers',
            name='Total System Load',
            line=dict(color='purple', width=3),
            marker=dict(size=4),
            hovertemplate="<b>Total System Load</b><br>" +
                         "Time: %{x}<br>" +
                         "Load: %{y:,.2f} kWh<br>" +
                         "<extra></extra>"
        ))
        
        aggregated_fig.update_layout(
            title=f"Aggregated System Load Prediction - {current_date}",
            xaxis_title="Time",
            yaxis_title="Total System Load (kWh)",
            template="plotly_white",
            hovermode="x unified",
            width=1200,
            height=600,
            showlegend=True
        )
        
        # Add peak and minimum annotations
        peak_idx = aggregated_df['Load_All'].idxmax()
        min_idx = aggregated_df['Load_All'].idxmin()
        
        peak_time = aggregated_df.loc[peak_idx, 'TradeDateTime']
        peak_load = aggregated_df.loc[peak_idx, 'Load_All']
        min_time = aggregated_df.loc[min_idx, 'TradeDateTime']
        min_load = aggregated_df.loc[min_idx, 'Load_All']
        
        aggregated_fig.add_annotation(
            x=peak_time,
            y=peak_load,
            text=f"Peak: {peak_load:,.0f} kWh",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            arrowwidth=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="red"
        )
        
        aggregated_fig.add_annotation(
            x=min_time,
            y=min_load,
            text=f"Min: {min_load:,.0f} kWh",
            showarrow=True,
            arrowhead=2,
            arrowcolor="blue",
            arrowwidth=2,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="blue"
        )
        
        # Save aggregated plot
        aggregated_plot_file = os.path.join(output_path, f"aggregated_predictions_{current_date}.html")
        aggregated_fig.write_html(aggregated_plot_file)
        
        logger.info("Aggregated plot created and saved")
        
    except Exception as e:
        logger.error(f"Aggregated plot creation failed: {str(e)}")

def _save_results_to_files(combined_df, aggregated_df, output_path, current_date):
    """Save combined and aggregated results to CSV files"""
    try:
        # Save combined results
        combined_file = os.path.join(output_path, f"Combined_Load_{current_date}.csv")
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Combined results saved: {combined_file}")
        
        # Save aggregated results
        aggregated_file = os.path.join(output_path, f"Aggregated_Load_{current_date}.csv")
        aggregated_df.to_csv(aggregated_file, index=False)
        logger.info(f"Aggregated results saved: {aggregated_file}")
        
        # Create summary statistics
        summary_stats = {
            'generation_time': datetime.now().isoformat(),
            'forecast_date': current_date,
            'total_profiles': len(combined_df['Profile'].unique()),
            'total_hours': len(aggregated_df),
            'total_system_load_kwh': float(aggregated_df['Load_All'].sum()),
            'peak_system_load_kwh': float(aggregated_df['Load_All'].max()),
            'peak_hour': int(aggregated_df.loc[aggregated_df['Load_All'].idxmax(), 'Hour']),
            'min_system_load_kwh': float(aggregated_df['Load_All'].min()),
            'min_hour': int(aggregated_df.loc[aggregated_df['Load_All'].idxmin(), 'Hour']),
            'avg_hourly_load_kwh': float(aggregated_df['Load_All'].mean()),
            'profile_breakdown': {}
        }
        
        # Add profile breakdown
        for profile in combined_df['Profile'].unique():
            profile_data = combined_df[combined_df['Profile'] == profile]
            summary_stats['profile_breakdown'][profile] = {
                'total_load_kwh': float(profile_data['Load_All'].sum()),
                'peak_load_kwh': float(profile_data['Load_All'].max()),
                'avg_load_kwh': float(profile_data['Load_All'].mean()),
                'records': len(profile_data)
            }
        
        # Save summary statistics
        summary_file = os.path.join(output_path, f"forecast_summary_{current_date}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        logger.info(f"Summary statistics saved: {summary_file}")
        
    except Exception as e:
        logger.error(f"Results saving failed: {str(e)}")
