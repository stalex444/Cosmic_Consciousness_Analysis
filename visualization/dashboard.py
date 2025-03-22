#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Dashboard
----------------------
Interactive dashboard for visualizing cosmic consciousness analysis results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd

# Try to import interactive visualization libraries
try:
    # For Python 2.7 compatibility with Dash
    import dash
    if hasattr(dash, 'dash'):
        # Old-style import for Python 2.7
        import dash.dcc as dcc
        import dash.html as html
        from dash.dependencies import Input, Output
    else:
        # New-style imports will fail in Python 2.7
        from dash import dcc, html
        from dash.dependencies import Input, Output
    import plotly
    # Check plotly version for compatibility
    if int(plotly.__version__.split('.')[0]) >= 5:
        import plotly.express as px
        import plotly.graph_objects as go
    else:
        # For older plotly versions
        import plotly.graph_objs as go
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False


class VisualizationDashboard(object):  # Use object as base class for Python 2.7
    """
    Dashboard for visualizing cosmic consciousness analysis results.
    
    This class provides both static and interactive visualizations of the
    analysis results, making it easier to interpret and communicate findings.
    """
    
    def __init__(self, results_dir):
        """
        Initialize the dashboard with results directory.
        
        Args:
            results_dir (str): Directory containing analysis results
        """
        self.results_dir = results_dir
        self.results = None
        self.load_results()
        
        # Set color scheme
        self.colors = {
            'phi': '#FFD700',  # Gold
            'pi': '#4682B4',   # Steel Blue
            'e': '#2E8B57',    # Sea Green
            'sqrt2': '#8B4513',  # Saddle Brown
            'sqrt3': '#9932CC',  # Dark Orchid
            'sqrt5': '#B22222',  # Firebrick
            'background': '#F5F5F5',  # Light gray
            'text': '#333333',  # Dark gray
            'highlight': '#FF5733',  # Coral
            'significant': '#228B22',  # Forest Green
            'not_significant': '#B22222',  # Firebrick
        }
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def load_results(self):
        """
        Load analysis results from JSON file.
        """
        results_file = os.path.join(self.results_dir, "results.json")
        
        if not os.path.exists(results_file):
            print("Results file not found: {}".format(results_file))
            return False
        
        try:
            with open(results_file, 'r') as f:
                self.results = json.load(f)
            print("Results loaded from {}".format(results_file))
            return True
        except Exception as e:
            print("Error loading results: {}".format(e))
            return False
    
    def create_static_dashboard(self, output_file=None):
        """
        Create a static dashboard with multiple visualizations.
        
        Args:
            output_file (str, optional): Output file path. If None, the dashboard
                will be saved to the results directory.
        
        Returns:
            str: Path to the saved dashboard image
        """
        if self.results is None:
            print("No results loaded. Call load_results() first.")
            return None
        
        # Set output file
        if output_file is None:
            output_file = os.path.join(self.results_dir, "dashboard.png")
        
        # Create figure
        fig = plt.figure(figsize=(16, 12), facecolor=self.colors['background'])
        gs = GridSpec(3, 4, figure=fig)
        
        # Add title
        fig.suptitle("Cosmic Consciousness Analysis Dashboard", 
                     fontsize=20, fontweight='bold', color=self.colors['text'],
                     y=0.98)
        
        # Add timestamp
        fig.text(0.5, 0.955, "Analysis completed on: {}".format(self.results['timestamp']),
                 ha='center', fontsize=12, color=self.colors['text'])
        
        # 1. Phi Optimality by Test (Top Left)
        ax1 = fig.add_subplot(gs[0, 0:2])
        self._plot_phi_optimality(ax1)
        
        # 2. P-values by Test (Top Right)
        ax2 = fig.add_subplot(gs[0, 2:4])
        self._plot_p_values(ax2)
        
        # 3. Best Constants Distribution (Middle Left)
        ax3 = fig.add_subplot(gs[1, 0:2])
        self._plot_best_constants(ax3)
        
        # 4. Combined Significance (Middle Right)
        ax4 = fig.add_subplot(gs[1, 2:4])
        self._plot_combined_significance(ax4)
        
        # 5. Test Results Heatmap (Bottom)
        ax5 = fig.add_subplot(gs[2, 0:4])
        self._plot_test_results_heatmap(ax5)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Dashboard saved to {}".format(output_file))
        return output_file
    
    def _plot_phi_optimality(self, ax):
        """Plot phi optimality by test."""
        phi_optimality = self.results['combined']['phi_optimality']
        test_names = list(phi_optimality.keys())
        
        # Extract values, handling the new dictionary structure
        values = []
        for test_name, phi_value in phi_optimality.items():
            if isinstance(phi_value, dict):
                # If it's a dictionary of dictionaries (from our fix)
                if all(isinstance(v, dict) for v in phi_value.values()):
                    # Calculate average phi optimality across all ratios
                    phi_values = []
                    for ratio_dict in phi_value.values():
                        if 'phi' in ratio_dict:
                            phi_values.append(ratio_dict['phi'])
                    
                    if phi_values:
                        values.append(float(np.mean(phi_values)))
                    else:
                        values.append(0.0)  # Default if no phi values found
                # If it's a simple dictionary with 'phi' key
                elif 'phi' in phi_value:
                    values.append(float(phi_value['phi']))
                else:
                    values.append(0.0)  # Default if no phi value found
            else:
                # Original case - direct float value
                values.append(float(phi_value))
        
        # Create bar chart
        bars = ax.bar(range(len(test_names)), values, color=self.colors['phi'])
        
        # Add labels
        ax.set_xticks(range(len(test_names)))
        ax.set_xticklabels([name.replace(' Test', '') for name in test_names], 
                          rotation=45, ha='right')
        ax.set_ylabel('Phi Optimality')
        ax.set_title('Phi Optimality by Test')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   '{:.3f}'.format(height), ha='center', va='bottom', 
                   fontsize=9, color=self.colors['text'])
    
    def _plot_p_values(self, ax):
        """Plot p-values by test."""
        p_values = self.results['combined']['p_values']
        test_names = list(p_values.keys())
        values = list(p_values.values())
        
        # Create bar chart with color based on significance
        bars = ax.bar(range(len(test_names)), values, 
                     color=[self.colors['significant'] if v < 0.05 else 
                           self.colors['not_significant'] for v in values])
        
        # Add labels
        ax.set_xticks(range(len(test_names)))
        ax.set_xticklabels([name.replace(' Test', '') for name in test_names], 
                          rotation=45, ha='right')
        ax.set_ylabel('P-value')
        ax.set_title('P-values by Test')
        
        # Add significance threshold
        ax.axhline(0.05, color='black', linestyle='--', label='p=0.05')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   '{:.3f}'.format(height), ha='center', va='bottom', 
                   fontsize=9, color=self.colors['text'])
    
    def _plot_best_constants(self, ax):
        """Plot best constants distribution."""
        best_constants = self.results['combined']['best_constants']
        test_names = list(best_constants.keys())
        constants = [best_constants[name]['constant'] for name in test_names]
        
        # Count occurrences of each constant
        constant_counts = {}
        for constant in constants:
            if constant in constant_counts:
                constant_counts[constant] += 1
            else:
                constant_counts[constant] = 1
        
        # Plot pie chart
        constant_names = list(constant_counts.keys())
        constant_values = list(constant_counts.values())
        
        # Create colors for pie chart
        colors = [self.colors.get(c, 'gray') for c in constant_names]
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            constant_values, 
            labels=constant_names, 
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            wedgeprops={'edgecolor': 'w', 'linewidth': 1}
        )
        
        # Customize text
        for text in texts:
            text.set_color(self.colors['text'])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Best Constants Distribution')
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    def _plot_combined_significance(self, ax):
        """Plot combined significance."""
        if 'combined_significance' in self.results['combined']:
            significance = self.results['combined']['combined_significance']
            
            # Create a text box with the significance information
            if significance['significant']:
                text = "SIGNIFICANT EVIDENCE\nP-value: {:.6f}".format(significance['p_value'])
                box_color = self.colors['significant']
                alpha = 0.2
            else:
                text = "NO SIGNIFICANT EVIDENCE\nP-value: {:.6f}".format(significance['p_value'])
                box_color = self.colors['not_significant']
                alpha = 0.1
            
            # Add text box
            ax.text(0.5, 0.5, text, fontsize=14, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor=box_color, alpha=alpha))
            
            # Add title
            ax.set_title('Combined Significance')
            
            # Instead of using inset_axes, create a simple colored circle to indicate significance
            if significance['significant']:
                circle = plt.Circle((0.8, 0.8), 0.15, color=self.colors['significant'])
                ax.add_artist(circle)
                ax.text(0.8, 0.8, 'Sig.', ha='center', va='center', color='white', fontsize=10)
            else:
                circle = plt.Circle((0.8, 0.8), 0.15, color=self.colors['not_significant'])
                ax.add_artist(circle)
                ax.text(0.8, 0.8, 'Not Sig.', ha='center', va='center', color='black', fontsize=8)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _plot_test_results_heatmap(self, ax):
        """Plot test results heatmap."""
        # Extract test results
        test_results = {}
        for test_name, results in self.results['tests'].items():
            # Extract phi_optimality, handling the new dictionary structure
            phi_value = 0.0
            if 'phi_optimality' in results:
                phi_opt = results['phi_optimality']
                if isinstance(phi_opt, dict):
                    # If it's a dictionary of dictionaries
                    if all(isinstance(v, dict) for v in phi_opt.values()):
                        # Calculate average phi optimality across all ratios
                        phi_values = []
                        for ratio_dict in phi_opt.values():
                            if 'phi' in ratio_dict:
                                phi_values.append(ratio_dict['phi'])
                        
                        if phi_values:
                            phi_value = float(np.mean(phi_values))
                    # If it's a simple dictionary with 'phi' key
                    elif 'phi' in phi_opt:
                        phi_value = float(phi_opt['phi'])
                else:
                    # Original case - direct float value
                    phi_value = float(phi_opt)
            
            # Extract p_value
            p_value = float(results.get('p_value', 1.0))
            
            # Extract best value and constant
            best_value = 0.0
            best_constant = 'unknown'
            if 'best_constant' in results:
                best_const = results.get('best_constant', {})
                if isinstance(best_const, dict):
                    if 'value' in best_const:
                        best_value = float(best_const['value'])
                    if 'name' in best_const:
                        best_constant = best_const['name']
                else:
                    best_constant = str(best_const)
            
            test_results[test_name] = {
                'phi_optimality': phi_value,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'best_value': best_value,
                'best_constant': best_constant
            }
        
        # Create DataFrame
        df = pd.DataFrame.from_dict(test_results, orient='index')
        
        # Create metrics for heatmap
        metrics = ['phi_optimality', 'p_value', 'best_value']
        metric_names = ['Phi Optimality', 'P-value', 'Best Value']
        
        # Create data for heatmap
        heatmap_data = df[metrics].values.astype(float)  # Ensure all values are float
        
        # Create custom colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap=cmap, 
                   xticklabels=metric_names, yticklabels=df.index, ax=ax)
        
        # Add title
        ax.set_title('Test Results Summary')
        
        # Add color coding for best constant - only if we have best constants
        if 'best_constant' in df.columns:
            # Add color coding for best constant
            for i, (idx, row) in enumerate(df.iterrows()):
                constant = row['best_constant']
                color = self.colors.get(constant, 'gray')
                ax.add_patch(plt.Rectangle((3, i), 0.2, 1, color=color))
            
            # Add legend for constants
            handles = []
            for constant, color in self.colors.items():
                if constant in ['phi', 'pi', 'e', 'sqrt2', 'sqrt3', 'sqrt5']:
                    handles.append(mpatches.Patch(color=color, label=constant))
            
            # Only add legend if we have handles
            if handles:
                ax.legend(handles=handles, title='Best Constant', 
                         bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def create_interactive_dashboard(self, port=8050):
        """
        Create an interactive dashboard using Dash and Plotly.
        
        Args:
            port (int, optional): Port to run the dashboard on. Defaults to 8050.
        
        Returns:
            dash.Dash: Dash app
        """
        if not INTERACTIVE_AVAILABLE:
            print("Interactive dashboard requires dash and plotly. Install with:")
            print("pip install dash plotly")
            return None
        
        if self.results is None:
            print("No results loaded. Call load_results() first.")
            return None
        
        # Create Dash app
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1("Cosmic Consciousness Analysis Dashboard"),
            html.H4("Analysis completed on: {}".format(self.results['timestamp'])),
            
            html.Div([
                html.Div([
                    html.H3("Phi Optimality by Test"),
                    dcc.Graph(id='phi-optimality-graph')
                ], className='six columns'),
                
                html.Div([
                    html.H3("P-values by Test"),
                    dcc.Graph(id='p-values-graph')
                ], className='six columns'),
            ], className='row'),
            
            html.Div([
                html.Div([
                    html.H3("Best Constants Distribution"),
                    dcc.Graph(id='best-constants-graph')
                ], className='six columns'),
                
                html.Div([
                    html.H3("Combined Significance"),
                    dcc.Graph(id='combined-significance-graph')
                ], className='six columns'),
            ], className='row'),
            
            html.Div([
                html.H3("Test Results Summary"),
                dcc.Graph(id='test-results-heatmap')
            ], className='row'),
            
            html.Div([
                html.H3("Test Details"),
                dcc.Dropdown(
                    id='test-selector',
                    options=[{'label': test, 'value': test} 
                             for test in self.results['tests'].keys()],
                    value=list(self.results['tests'].keys())[0]
                ),
                html.Div(id='test-details')
            ], className='row')
        ])
        
        # Define callbacks
        @app.callback(
            Output('phi-optimality-graph', 'figure'),
            Input('test-selector', 'value')
        )
        def update_phi_optimality_graph(selected_test):
            phi_optimality = self.results['combined']['phi_optimality']
            test_names = list(phi_optimality.keys())
            values = list(phi_optimality.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[name.replace(' Test', '') for name in test_names],
                    y=values,
                    marker_color=self.colors['phi']
                )
            ])
            
            fig.update_layout(
                title='Phi Optimality by Test',
                xaxis_title='Test',
                yaxis_title='Phi Optimality',
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font_color=self.colors['text']
            )
            
            # Highlight selected test
            if selected_test in test_names:
                idx = test_names.index(selected_test)
                fig.add_shape(
                    type="rect",
                    x0=idx-0.4, y0=0,
                    x1=idx+0.4, y1=values[idx],
                    line=dict(color=self.colors['highlight'], width=3),
                    fillcolor="rgba(0,0,0,0)"
                )
            
            return fig
        
        @app.callback(
            Output('p-values-graph', 'figure'),
            Input('test-selector', 'value')
        )
        def update_p_values_graph(selected_test):
            p_values = self.results['combined']['p_values']
            test_names = list(p_values.keys())
            values = list(p_values.values())
            
            # Create colors based on significance
            colors = [self.colors['significant'] if v < 0.05 else 
                     self.colors['not_significant'] for v in values]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=[name.replace(' Test', '') for name in test_names],
                    y=values,
                    marker_color=colors
                )
            ])
            
            # Add significance threshold
            fig.add_shape(
                type="line",
                x0=-0.5, y0=0.05,
                x1=len(test_names)-0.5, y1=0.05,
                line=dict(color="black", width=2, dash="dash"),
            )
            
            fig.update_layout(
                title='P-values by Test',
                xaxis_title='Test',
                yaxis_title='P-value',
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font_color=self.colors['text']
            )
            
            # Highlight selected test
            if selected_test in test_names:
                idx = test_names.index(selected_test)
                fig.add_shape(
                    type="rect",
                    x0=idx-0.4, y0=0,
                    x1=idx+0.4, y1=values[idx],
                    line=dict(color=self.colors['highlight'], width=3),
                    fillcolor="rgba(0,0,0,0)"
                )
            
            return fig
        
        @app.callback(
            Output('best-constants-graph', 'figure'),
            Input('test-selector', 'value')
        )
        def update_best_constants_graph(selected_test):
            best_constants = self.results['combined']['best_constants']
            test_names = list(best_constants.keys())
            constants = [best_constants[name]['constant'] for name in test_names]
            
            # Count occurrences of each constant
            constant_counts = {}
            for constant in constants:
                if constant in constant_counts:
                    constant_counts[constant] += 1
                else:
                    constant_counts[constant] = 1
            
            # Create pie chart
            constant_names = list(constant_counts.keys())
            constant_values = list(constant_counts.values())
            
            # Create colors for pie chart
            colors = [self.colors.get(c, 'gray') for c in constant_names]
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=constant_names,
                    values=constant_values,
                    marker_colors=colors
                )
            ])
            
            fig.update_layout(
                title='Best Constants Distribution',
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font_color=self.colors['text']
            )
            
            # Highlight selected test's constant
            if selected_test in test_names:
                selected_constant = best_constants[selected_test]['constant']
                fig.add_annotation(
                    text="Selected test's constant: {}".format(selected_constant),
                    x=0.5, y=1.1,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(color=self.colors.get(selected_constant, 'gray'), size=14)
                )
            
            return fig
        
        @app.callback(
            Output('test-details', 'children'),
            Input('test-selector', 'value')
        )
        def update_test_details(selected_test):
            if selected_test not in self.results['tests']:
                return html.Div("Test not found")
            
            test_results = self.results['tests'][selected_test]
            
            # Create details table
            rows = []
            for key, value in test_results.items():
                if key in ['data', 'surrogate_data']:
                    continue
                
                if isinstance(value, dict):
                    rows.append(html.Tr([
                        html.Td(key),
                        html.Td(str(value))
                    ]))
                elif isinstance(value, list):
                    if len(value) > 5:
                        rows.append(html.Tr([
                            html.Td(key),
                            html.Td(str(value[:5]) + "...")
                        ]))
                    else:
                        rows.append(html.Tr([
                            html.Td(key),
                            html.Td(str(value))
                        ]))
                else:
                    rows.append(html.Tr([
                        html.Td(key),
                        html.Td(str(value))
                    ]))
            
            return html.Table(
                [html.Tr([html.Th("Metric"), html.Th("Value")])] + rows,
                className='table'
            )
        
        # Run server
        print("Starting dashboard server on port {}...".format(port))
        app.run_server(debug=True, port=port)
        
        return app


def main():
    """Run the dashboard with sample results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run visualization dashboard")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing analysis results")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Output file for static dashboard")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive dashboard")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port for interactive dashboard")
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = VisualizationDashboard(args.results_dir)
    
    # Create static dashboard
    if not args.interactive:
        dashboard.create_static_dashboard(args.output_file)
    else:
        if not INTERACTIVE_AVAILABLE:
            print("Interactive dashboard requires dash and plotly. Install with:")
            print("pip install dash==1.19.0 plotly==4.14.3")
            return
        
        dashboard.create_interactive_dashboard(args.port)


if __name__ == "__main__":
    main()
