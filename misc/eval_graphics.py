#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import argparse
from matplotlib.gridspec import GridSpec
from datetime import datetime

class PromptOptimizationAnalyzer:
    """Class to analyze prompt optimization results"""
    
    def __init__(self, base_dir: str = 'results', output_dir: str = 'analysis_results'):
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.results = {}
        self.csv_files = []
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'comparison_plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'dashboard'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'data'), exist_ok=True)
        
        # Set plot style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
        })
    
    def find_csv_files(self) -> List[Dict]:
        """Find all step_results_eval.csv files in the directory structure"""
        print(f"Scanning {self.base_dir} for CSV files...")
        csv_files = []
        
        try:
            # Traverse directory structure
            for dataset in os.listdir(self.base_dir):
                dataset_path = os.path.join(self.base_dir, dataset)
                if not os.path.isdir(dataset_path):
                    continue
                    
                for model in os.listdir(dataset_path):
                    model_path = os.path.join(dataset_path, model)
                    if not os.path.isdir(model_path):
                        continue
                        
                    for optimizer in os.listdir(model_path):
                        optimizer_path = os.path.join(model_path, optimizer)
                        if not os.path.isdir(optimizer_path):
                            continue
                            
                        for seed in os.listdir(optimizer_path):
                            seed_path = os.path.join(optimizer_path, seed)
                            if not os.path.isdir(seed_path):
                                continue
                                
                            csv_path = os.path.join(seed_path, 'step_results_eval.csv')
                            if os.path.isfile(csv_path):
                                csv_files.append({
                                    'dataset': dataset,
                                    'model': model,
                                    'optimizer': optimizer,
                                    'seed': seed,
                                    'path': csv_path
                                })
            
            print(f"Found {len(csv_files)} CSV files")
            self.csv_files = csv_files
            return csv_files
            
        except Exception as e:
            print(f"Error finding CSV files: {e}")
            return []
    
    def load_and_process_csv_files(self) -> Dict:
        """Load and process CSV files, grouping by dataset-model-optimizer"""
        if not self.csv_files:
            print("No CSV files found.")
            return {}
        
        print("Processing CSV files...")
        
        # Group files by dataset-model-optimizer
        grouped_files = {}
        for file in self.csv_files:
            key = f"{file['dataset']}-{file['model']}-{file['optimizer']}"
            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file)
        
        results = {}
        
        # Process each combination
        for key, files in grouped_files.items():
            print(f"Processing {key}...")
            
            # Initialize result object with metadata
            results[key] = {
                'dataset': files[0]['dataset'],
                'model': files[0]['model'],
                'optimizer': files[0]['optimizer'],
                'seed_data': {},
                'seeds': [file['seed'] for file in files],
                'num_seeds': len(files)
            }
            
            # Read data from each seed
            for file in files:
                try:
                    df = pd.read_csv(file['path'])
                    results[key]['seed_data'][file['seed']] = df
                except Exception as e:
                    print(f"Error reading {file['path']}: {e}")
            
            # Calculate statistics
            self._calculate_statistics(results[key])
        
        self.results = results
        
        # Save processed data
        self._save_processed_data()
        
        return results
    
    def _calculate_statistics(self, combination_data: Dict) -> None:
        """Calculate mean and standard deviation for each step"""
        seed_data = combination_data['seed_data']
        if not seed_data:
            combination_data['stats'] = []
            return
        
        # Get all unique steps
        all_steps = set()
        for seed, df in seed_data.items():
            all_steps.update(df['step'].unique())
        
        sorted_steps = sorted(all_steps)
        
        # Calculate statistics for each step
        stats = []
        for step in sorted_steps:
            # Collect test_scores for this step across all seeds
            scores = []
            prompts = []
            
            for seed, df in seed_data.items():
                step_data = df[df['step'] == step]
                if not step_data.empty:
                    scores.append(step_data['test_score'].iloc[0])
                    if 'prompt' in step_data.columns:
                        prompts.append(step_data['prompt'].iloc[0])
            
            if scores:
                mean = np.mean(scores)
                std = np.std(scores, ddof=1)  # Use n-1 for sample standard deviation
                
                stat_entry = {
                    'step': step,
                    'mean': mean,
                    'std': std,
                    'scores': scores,
                    'min': mean - std,
                    'max': mean + std,
                    'min_score': min(scores),
                    'max_score': max(scores),
                    'score_range': max(scores) - min(scores)
                }
                
                # Store representative prompt if available
                if prompts:
                    stat_entry['prompt'] = prompts[0]  # Just pick the first one
                
                stats.append(stat_entry)
        
        combination_data['stats'] = stats
        
        # Calculate additional metrics
        if stats:
            # Find best performance
            best_idx = np.argmax([stat['mean'] for stat in stats])
            combination_data['best_performance'] = stats[best_idx]
            
            # Calculate convergence (when we reach 95% of best performance)
            best_score = stats[best_idx]['mean']
            threshold = 0.95 * best_score
            convergence_step = None
            
            for stat in stats:
                if stat['mean'] >= threshold:
                    convergence_step = stat['step']
                    break
            
            combination_data['convergence_step'] = convergence_step
            
            # Calculate improvement over initial performance
            if stats[0]['mean'] > 0:  # Avoid division by zero
                improvement = (best_score - stats[0]['mean']) / stats[0]['mean'] * 100
            else:
                improvement = np.nan
                
            combination_data['improvement_percentage'] = improvement
    
    def _save_processed_data(self) -> None:
        """Save processed data to JSON for later use"""
        output_file = os.path.join(self.output_dir, 'data', 'processed_results.json')
        
        # Convert data to JSON-serializable format
        json_data = {}
        for key, data in self.results.items():
            json_data[key] = {
                'dataset': data['dataset'],
                'model': data['model'],
                'optimizer': data['optimizer'],
                'num_seeds': data['num_seeds'],
                'seeds': data['seeds'],
                'stats': []
            }
            
            for stat in data['stats']:
                json_stat = {
                    'step': stat['step'],
                    'mean': stat['mean'],
                    'std': stat['std'],
                    'min': stat['min'],
                    'max': stat['max'],
                    'min_score': stat['min_score'],
                    'max_score': stat['max_score'],
                    'score_range': stat['score_range']
                }
                json_data[key]['stats'].append(json_stat)
            
            if 'best_performance' in data:
                json_data[key]['best_performance'] = {
                    'step': data['best_performance']['step'],
                    'mean': data['best_performance']['mean'],
                    'std': data['best_performance']['std']
                }
            
            if 'convergence_step' in data:
                json_data[key]['convergence_step'] = data['convergence_step']
                
            if 'improvement_percentage' in data:
                json_data[key]['improvement_percentage'] = data['improvement_percentage']
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved processed data to {output_file}")
    
    def create_individual_plots(self) -> None:
        """Create score vs steps plots for each dataset-model-optimizer combination"""
        if not self.results:
            print("No results available. Run load_and_process_csv_files first.")
            return
        
        print("Creating individual plots...")
        output_dir = os.path.join(self.output_dir, 'plots')
        
        for key, data in self.results.items():
            if not data['stats']:
                print(f"No statistics available for {key}, skipping.")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract data for plotting
            steps = [stat['step'] for stat in data['stats']]
            means = [stat['mean'] for stat in data['stats']]
            mins = [stat['min'] for stat in data['stats']]
            maxs = [stat['max'] for stat in data['stats']]
            
            # Plot mean line
            ax.plot(steps, means, marker='o', markersize=4, linewidth=2, label='Mean Score')
            
            # Plot std dev area
            ax.fill_between(steps, mins, maxs, alpha=0.2, label='±1 Std Dev')
            
            # Set labels and title
            ax.set_xlabel('Step')
            ax.set_ylabel('Test Score')
            ax.set_title(f"Score vs Steps: {data['dataset']} - {data['model']} - {data['optimizer']}")
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            output_path = os.path.join(output_dir, f"{key}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            
            print(f"Created plot: {output_path}")
    
    def create_comparison_plots(self) -> None:
        """Create comparison plots between CAPO and EvoPromptGA for each dataset-model pair"""
        if not self.results:
            print("No results available. Run load_and_process_csv_files first.")
            return
        
        print("Creating comparison plots...")
        output_dir = os.path.join(self.output_dir, 'comparison_plots')
        
        # Group by dataset and model
        dataset_model_pairs = {}
        for key, data in self.results.items():
            pair_key = f"{data['dataset']}-{data['model']}"
            if pair_key not in dataset_model_pairs:
                dataset_model_pairs[pair_key] = {}
            dataset_model_pairs[pair_key][data['optimizer']] = data
        
        # Create comparison plots
        for pair_key, optimizers in dataset_model_pairs.items():
            if len(optimizers) < 2:
                print(f"Not enough optimizers for comparison in {pair_key}, skipping.")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 7))
            
            colors = {'capo': 'blue', 'EvoPromptGA': 'red'}
            markers = {'capo': 'o', 'EvoPromptGA': 's'}
            
            # Plot data for each optimizer
            for optimizer_name, color in colors.items():
                if optimizer_name not in optimizers:
                    continue
                
                optimizer_data = optimizers[optimizer_name]
                steps = [stat['step'] for stat in optimizer_data['stats']]
                means = [stat['mean'] for stat in optimizer_data['stats']]
                mins = [stat['min'] for stat in optimizer_data['stats']]
                maxs = [stat['max'] for stat in optimizer_data['stats']]
                
                ax.plot(steps, means, color=color, marker=markers[optimizer_name], 
                        markersize=5, linewidth=2, label=f"{optimizer_name} (Mean)")
                ax.fill_between(steps, mins, maxs, color=color, alpha=0.15)
            
            # Set labels and title
            dataset, model = pair_key.split('-')
            ax.set_xlabel('Step')
            ax.set_ylabel('Test Score')
            ax.set_title(f"Optimizer Comparison: {dataset} - {model}")
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            output_path = os.path.join(output_dir, f"comparison_{pair_key}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            
            print(f"Created comparison plot: {output_path}")
    
    def create_model_comparison_plots(self) -> None:
        """Create plots comparing different models for each dataset-optimizer pair"""
        if not self.results:
            print("No results available. Run load_and_process_csv_files first.")
            return
        
        print("Creating model comparison plots...")
        output_dir = os.path.join(self.output_dir, 'comparison_plots')
        
        # Group by dataset and optimizer
        dataset_optimizer_pairs = {}
        for key, data in self.results.items():
            pair_key = f"{data['dataset']}-{data['optimizer']}"
            if pair_key not in dataset_optimizer_pairs:
                dataset_optimizer_pairs[pair_key] = {}
            dataset_optimizer_pairs[pair_key][data['model']] = data
        
        # Create comparison plots
        for pair_key, models in dataset_optimizer_pairs.items():
            if len(models) < 2:
                print(f"Not enough models for comparison in {pair_key}, skipping.")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Define colors for models
            model_colors = {
                'qwen': '#1f77b4',    # blue
                'llama': '#ff7f0e',   # orange
                'mistral': '#2ca02c'  # green
            }
            
            model_markers = {
                'qwen': 'o',
                'llama': 's',
                'mistral': '^'
            }
            
            # Plot data for each model
            for model_name, model_data in models.items():
                color = model_colors.get(model_name, 'gray')
                marker = model_markers.get(model_name, 'o')
                
                steps = [stat['step'] for stat in model_data['stats']]
                means = [stat['mean'] for stat in model_data['stats']]
                mins = [stat['min'] for stat in model_data['stats']]
                maxs = [stat['max'] for stat in model_data['stats']]
                
                ax.plot(steps, means, color=color, marker=marker, 
                        markersize=5, linewidth=2, label=f"{model_name}")
                ax.fill_between(steps, mins, maxs, color=color, alpha=0.15)
            
            # Set labels and title
            dataset, optimizer = pair_key.split('-')
            ax.set_xlabel('Step')
            ax.set_ylabel('Test Score')
            ax.set_title(f"Model Comparison: {dataset} - {optimizer}")
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Save figure
            output_path = os.path.join(output_dir, f"model_comparison_{pair_key}.png")
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close(fig)
            
            print(f"Created model comparison plot: {output_path}")
    
    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of results"""
        if not self.results:
            print("No results available. Run load_and_process_csv_files first.")
            return pd.DataFrame()
        
        print("Creating summary table...")
        
        # Prepare summary data
        summary_data = []
        
        for key, data in self.results.items():
            if not data['stats']:
                continue
            
            # Get initial, final, and best scores
            initial_score = data['stats'][0]['mean'] if data['stats'] else np.nan
            final_score = data['stats'][-1]['mean'] if data['stats'] else np.nan
            
            best_score = np.nan
            best_step = np.nan
            convergence_step = np.nan
            improvement_pct = np.nan
            
            if 'best_performance' in data:
                best_score = data['best_performance']['mean']
                best_step = data['best_performance']['step']
            
            if 'convergence_step' in data:
                convergence_step = data['convergence_step']
                
            if 'improvement_percentage' in data:
                improvement_pct = data['improvement_percentage']
            
            summary_data.append({
                'dataset': data['dataset'],
                'model': data['model'],
                'optimizer': data['optimizer'],
                'num_seeds': data['num_seeds'],
                'initial_score': initial_score,
                'final_score': final_score,
                'best_score': best_score,
                'best_step': best_step,
                'convergence_step': convergence_step,
                'improvement_pct': improvement_pct
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Round numeric columns
        numeric_cols = ['initial_score', 'final_score', 'best_score', 'improvement_pct']
        summary_df[numeric_cols] = summary_df[numeric_cols].round(4)
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, 'data', 'results_summary.csv')
        summary_df.to_csv(output_file, index=False)
        print(f"Saved summary table to {output_file}")
        
        return summary_df
    
    def create_dashboard(self) -> None:
        """Create a comprehensive dashboard with key insights"""
        if not self.results:
            print("No results available. Run load_and_process_csv_files first.")
            return
        
        print("Creating dashboard...")
        
        # Get summary data
        summary_df = self.create_summary_table()
        if summary_df.empty:
            print("No summary data available.")
            return
        
        # Create dashboard plot
        fig = plt.figure(figsize=(16, 20))
        gs = GridSpec(6, 2, figure=fig)
        
        # Title
        fig.suptitle(f"Prompt Optimization Results Dashboard\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                     fontsize=16, y=0.98)
        
        # 1. Top optimizers by average best score
        ax1 = fig.add_subplot(gs[0, 0])
        optimizer_scores = summary_df.groupby('optimizer')['best_score'].mean().reset_index()
        optimizer_scores = optimizer_scores.sort_values('best_score', ascending=False)
        
        sns.barplot(x='optimizer', y='best_score', data=optimizer_scores, ax=ax1)
        ax1.set_title('Optimizers by Avg. Best Score')
        ax1.set_xlabel('')
        ax1.set_ylabel('Avg. Best Score')
        for i, v in enumerate(optimizer_scores['best_score']):
            ax1.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        # 2. Top models by average best score
        ax2 = fig.add_subplot(gs[0, 1])
        model_scores = summary_df.groupby('model')['best_score'].mean().reset_index()
        model_scores = model_scores.sort_values('best_score', ascending=False)
        
        sns.barplot(x='model', y='best_score', data=model_scores, ax=ax2)
        ax2.set_title('Models by Avg. Best Score')
        ax2.set_xlabel('')
        ax2.set_ylabel('Avg. Best Score')
        for i, v in enumerate(model_scores['best_score']):
            ax2.text(i, v + 0.01, f"{v:.4f}", ha='center')
        
        # 3. Average improvement percentage by optimizer
        ax3 = fig.add_subplot(gs[1, 0])
        optimizer_improvement = summary_df.groupby('optimizer')['improvement_pct'].mean().reset_index()
        optimizer_improvement = optimizer_improvement.sort_values('improvement_pct', ascending=False)
        
        sns.barplot(x='optimizer', y='improvement_pct', data=optimizer_improvement, ax=ax3)
        ax3.set_title('Optimizers by Avg. Improvement %')
        ax3.set_xlabel('')
        ax3.set_ylabel('Avg. Improvement %')
        for i, v in enumerate(optimizer_improvement['improvement_pct']):
            ax3.text(i, v + 0.5, f"{v:.2f}%", ha='center')
        
        # 4. Average convergence step by optimizer
        ax4 = fig.add_subplot(gs[1, 1])
        optimizer_convergence = summary_df.groupby('optimizer')['convergence_step'].mean().reset_index()
        optimizer_convergence = optimizer_convergence.sort_values('convergence_step', ascending=True)
        
        sns.barplot(x='optimizer', y='convergence_step', data=optimizer_convergence, ax=ax4)
        ax4.set_title('Optimizers by Avg. Convergence Step')
        ax4.set_xlabel('')
        ax4.set_ylabel('Avg. Convergence Step')
        for i, v in enumerate(optimizer_convergence['convergence_step']):
            ax4.text(i, v + 0.2, f"{v:.1f}", ha='center')
        
        # 5. Heat map of best scores by dataset-model-optimizer
        ax5 = fig.add_subplot(gs[2:4, :])
        pivot_data = summary_df.pivot_table(
            index=['dataset', 'model'], 
            columns='optimizer', 
            values='best_score',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".4f", ax=ax5)
        ax5.set_title('Best Scores by Dataset-Model-Optimizer Combination')
        ax5.set_ylabel('')
        ax5.set_xlabel('')
        
        # 6. Box plot of best scores by optimizer
        ax6 = fig.add_subplot(gs[4, 0])
        sns.boxplot(x='optimizer', y='best_score', data=summary_df, ax=ax6)
        ax6.set_title('Distribution of Best Scores by Optimizer')
        ax6.set_xlabel('')
        ax6.set_ylabel('Best Score')
        
        # 7. Box plot of best scores by model
        ax7 = fig.add_subplot(gs[4, 1])
        sns.boxplot(x='model', y='best_score', data=summary_df, ax=ax7)
        ax7.set_title('Distribution of Best Scores by Model')
        ax7.set_xlabel('')
        ax7.set_ylabel('Best Score')
        
        # 8. Scatter plot of improvement vs convergence
        ax8 = fig.add_subplot(gs[5, :])
        scatter_data = summary_df.dropna(subset=['improvement_pct', 'convergence_step'])
        
        sns.scatterplot(
            x='convergence_step', 
            y='improvement_pct', 
            hue='optimizer', 
            style='model',
            s=100,
            data=scatter_data,
            ax=ax8
        )
        
        ax8.set_title('Improvement Percentage vs Convergence Step')
        ax8.set_xlabel('Convergence Step')
        ax8.set_ylabel('Improvement %')
        
        # Add a legend
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save dashboard
        output_path = os.path.join(self.output_dir, 'dashboard', 'results_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Created dashboard: {output_path}")
        
        # Also create an HTML report
        self._create_html_report(summary_df)
    
    def _create_html_report(self, summary_df: pd.DataFrame) -> None:
        """Create an HTML report of the results"""
        output_path = os.path.join(self.output_dir, 'dashboard', 'results_report.html')
        
        # Convert summary dataframe to HTML table
        summary_table = summary_df.to_html(classes='table table-striped', index=False)
        
        # Prepare HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prompt Optimization Results</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding: 20px; }}
                .plot-img {{ max-width: 100%; height: auto; margin-bottom: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                h1, h2, h3 {{ margin-top: 30px; }}
                .table-container {{ margin: 30px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="text-center mb-4">Prompt Optimization Results Report</h1>
                <p class="text-center text-muted">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>Summary of Results</h2>
                <div class="table-container">
                    {summary_table}
                </div>
                
                <h2>Dashboard</h2>
                <img src="results_dashboard.png" alt="Results Dashboard" class="plot-img">
                
                <h2>Optimizer Comparisons</h2>
                <div class="row">
        """
        
        # Add optimizer comparison plots
        comparison_dir = os.path.join(self.output_dir, 'comparison_plots')
        comparison_plots = [f for f in os.listdir(comparison_dir) if f.startswith('comparison_')]
        
        for plot in comparison_plots:
            html_content += f"""
                    <div class="col-md-6 mb-4">
                        <img src="../comparison_plots/{plot}" alt="{plot}" class="plot-img">
                    </div>
            """
        
        # Add model comparison plots
        html_content += """
                </div>
                
                <h2>Model Comparisons</h2>
                <div class="row">
        """
        
        model_plots = [f for f in os.listdir(comparison_dir) if f.startswith('model_comparison_')]
        
        for plot in model_plots:
            html_content += f"""
                    <div class="col-md-6 mb-4">
                        <img src="../comparison_plots/{plot}" alt="{plot}" class="plot-img">
                    </div>
            """
        
        # Add individual plots
        html_content += """
                </div>
                
                <h2>Individual Plots</h2>
                <div class="row">
        """
        
        plots_dir = os.path.join(self.output_dir, 'plots')
        individual_plots = os.listdir(plots_dir)
        
        for plot in individual_plots:
            html_content += f"""
                    <div class="col-md-4 mb-4">
                        <img src="../plots/{plot}" alt="{plot}" class="plot-img">
                    </div>
            """
        
        # Close HTML
        html_content += """
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Created HTML report: {output_path}")
    
    def run_analysis(self) -> None:
        """Run the full analysis pipeline"""
        print(f"Starting analysis of prompt optimization results in {self.base_dir}")
        
        # Find CSV files
        self.find_csv_files()
        
        if not self.csv_files:
            print("No CSV files found. Cannot proceed with analysis.")
            return
        
        # Load and process CSV files
        self.load_and_process_csv_files()
        
        # Create plots
        self.create_individual_plots()
        self.create_comparison_plots()
        self.create_model_comparison_plots()
        
        # Create dashboard and report
        self.create_dashboard()
        
        print(f"Analysis complete. Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze prompt optimization results')
    parser.add_argument('--base_dir', type=str, default='results', 
                        help='Base directory containing results (default: results)')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                        help='Output directory for analysis results (default: analysis_results)')
    
    args = parser.parse_args()
    
    analyzer = PromptOptimizationAnalyzer(base_dir=args.base_dir, output_dir=args.output_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()