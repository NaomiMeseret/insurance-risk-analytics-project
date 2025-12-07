"""
Comprehensive Exploratory Data Analysis for Insurance Risk Analytics

This module performs:
- Data summarization and descriptive statistics
- Data quality assessment
- Univariate and multivariate analysis
- Outlier detection
- Visualization of key insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data.load_data import load_insurance_data, get_data_info

# Set style for beautiful plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class InsuranceEDA:
    """Class for performing comprehensive EDA on insurance data"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize EDA with data loading.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the insurance data CSV file
        """
        self.df = load_insurance_data(data_path)
        self.reports_dir = Path(__file__).parent.parent.parent / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
    def data_summarization(self) -> pd.DataFrame:
        """
        Calculate descriptive statistics for numerical features.
        
        Returns:
        --------
        pd.DataFrame
            Descriptive statistics
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        desc_stats = self.df[numerical_cols].describe()
        
        # Add additional statistics
        desc_stats.loc['variance'] = self.df[numerical_cols].var()
        desc_stats.loc['skewness'] = self.df[numerical_cols].skew()
        desc_stats.loc['kurtosis'] = self.df[numerical_cols].kurtosis()
        
        return desc_stats
    
    def data_quality_assessment(self) -> dict:
        """
        Assess data quality including missing values and data types.
        
        Returns:
        --------
        dict
            Data quality metrics
        """
        info = get_data_info(self.df)
        
        quality_report = {
            'total_rows': info['shape'][0],
            'total_columns': info['shape'][1],
            'missing_values': info['missing_values'],
            'missing_percentage': info['missing_percentage'],
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage_mb': info['memory_usage_mb']
        }
        
        return quality_report
    
    def calculate_loss_ratio(self, groupby_cols: list = None) -> pd.DataFrame:
        """
        Calculate Loss Ratio (TotalClaims / TotalPremium).
        
        Parameters:
        -----------
        groupby_cols : list, optional
            Columns to group by (e.g., ['Province', 'VehicleType', 'Gender'])
        
        Returns:
        --------
        pd.DataFrame
            Loss ratio by groups
        """
        if groupby_cols:
            grouped = self.df.groupby(groupby_cols).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            }).reset_index()
        else:
            grouped = pd.DataFrame({
                'TotalClaims': [self.df['TotalClaims'].sum()],
                'TotalPremium': [self.df['TotalPremium'].sum()]
            })
        
        grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium']
        return grouped
    
    def plot_univariate_distributions(self, save: bool = True):
        """
        Plot histograms for numerical columns and bar charts for categorical columns.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plots
        """
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns[:10]  # Limit to 10
        categorical_cols = self.df.select_dtypes(include=['object']).columns[:10]  # Limit to 10
        
        # Numerical distributions
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                self.df[col].hist(bins=50, ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
        
        # Hide unused subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.reports_dir / 'univariate_numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Categorical distributions
        n_cols = 3
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(categorical_cols):
            if idx < len(axes):
                value_counts = self.df[col].value_counts().head(20)
                value_counts.plot(kind='bar', ax=axes[idx], edgecolor='black')
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                axes[idx].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.reports_dir / 'univariate_categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_bivariate_analysis(self, save: bool = True):
        """
        Explore relationships between TotalPremium and TotalClaims by ZipCode.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plots
        """
        # Scatter plot: TotalPremium vs TotalClaims
        plt.figure(figsize=(12, 8))
        plt.scatter(self.df['TotalPremium'], self.df['TotalClaims'], alpha=0.5, s=10)
        plt.xlabel('Total Premium')
        plt.ylabel('Total Claims')
        plt.title('Total Premium vs Total Claims')
        plt.grid(True, alpha=0.3)
        if save:
            plt.savefig(self.reports_dir / 'premium_vs_claims_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation matrix for key numerical variables
        key_vars = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
        available_vars = [v for v in key_vars if v in self.df.columns]
        
        if len(available_vars) > 1:
            corr_matrix = self.df[available_vars].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Matrix: Key Financial Variables')
            if save:
                plt.savefig(self.reports_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_geographic_trends(self, save: bool = True):
        """
        Compare trends over geography (Province, ZipCode).
        
        Parameters:
        -----------
        save : bool
            Whether to save the plots
        """
        if 'Province' in self.df.columns:
            # Loss ratio by Province
            loss_by_province = self.calculate_loss_ratio(['Province'])
            loss_by_province = loss_by_province.sort_values('LossRatio', ascending=False)
            
            plt.figure(figsize=(12, 6))
            plt.barh(loss_by_province['Province'], loss_by_province['LossRatio'])
            plt.xlabel('Loss Ratio')
            plt.ylabel('Province')
            plt.title('Loss Ratio by Province')
            plt.grid(True, alpha=0.3, axis='x')
            if save:
                plt.savefig(self.reports_dir / 'loss_ratio_by_province.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def detect_outliers(self, columns: list = None) -> dict:
        """
        Detect outliers using IQR method and box plots.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to check for outliers. If None, uses all numerical columns.
        
        Returns:
        --------
        dict
            Outlier information for each column
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_info = {}
        
        for col in columns:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(self.df) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return outlier_info
    
    def plot_outliers(self, columns: list = None, save: bool = True):
        """
        Create box plots to visualize outliers.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to plot. If None, uses key numerical columns.
        save : bool
            Whether to save the plots
        """
        if columns is None:
            key_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
            columns = [col for col in key_cols if col in self.df.columns]
        
        if not columns:
            return
        
        n_cols = min(2, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, col in enumerate(columns):
            if idx < len(axes):
                self.df.boxplot(column=col, ax=axes[idx])
                axes[idx].set_title(f'Box Plot: {col}')
                axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.reports_dir / 'outlier_detection_boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_trends(self, save: bool = True):
        """
        Analyze temporal trends in claims and premiums.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plots
        """
        if 'TransactionMonth' in self.df.columns:
            monthly_data = self.df.groupby('TransactionMonth').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'  # Claim frequency
            }).reset_index()
            
            monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Total Claims over time
            axes[0, 0].plot(monthly_data['TransactionMonth'], monthly_data['TotalClaims'], marker='o')
            axes[0, 0].set_title('Total Claims Over Time')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('Total Claims')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Total Premium over time
            axes[0, 1].plot(monthly_data['TransactionMonth'], monthly_data['TotalPremium'], marker='o', color='green')
            axes[0, 1].set_title('Total Premium Over Time')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Total Premium')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Loss Ratio over time
            axes[1, 0].plot(monthly_data['TransactionMonth'], monthly_data['LossRatio'], marker='o', color='red')
            axes[1, 0].set_title('Loss Ratio Over Time')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('Loss Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Claim frequency over time
            axes[1, 1].plot(monthly_data['TransactionMonth'], monthly_data['PolicyID'], marker='o', color='purple')
            axes[1, 1].set_title('Number of Policies Over Time')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Number of Policies')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            if save:
                plt.savefig(self.reports_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_creative_visualizations(self, save: bool = True):
        """
        Create 3 creative and beautiful plots capturing key insights.
        
        Parameters:
        -----------
        save : bool
            Whether to save the plots
        """
        # Visualization 1: Loss Ratio Heatmap by Province and Vehicle Type
        if 'Province' in self.df.columns and 'VehicleType' in self.df.columns:
            pivot_data = self.df.groupby(['Province', 'VehicleType']).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            }).reset_index()
            pivot_data['LossRatio'] = pivot_data['TotalClaims'] / pivot_data['TotalPremium']
            pivot_table = pivot_data.pivot(index='Province', columns='VehicleType', values='LossRatio')
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='RdYlGn_r', center=0.5,
                       linewidths=0.5, cbar_kws={'label': 'Loss Ratio'})
            plt.title('Loss Ratio Heatmap: Province vs Vehicle Type', fontsize=16, fontweight='bold')
            plt.xlabel('Vehicle Type', fontsize=12)
            plt.ylabel('Province', fontsize=12)
            plt.tight_layout()
            if save:
                plt.savefig(self.reports_dir / 'creative_viz1_loss_ratio_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualization 2: Risk Distribution by Gender with Premium and Claims
        if 'Gender' in self.df.columns:
            gender_analysis = self.df.groupby('Gender').agg({
                'TotalPremium': 'sum',
                'TotalClaims': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            gender_analysis['LossRatio'] = gender_analysis['TotalClaims'] / gender_analysis['TotalPremium']
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Bar chart: Premium and Claims by Gender
            x = np.arange(len(gender_analysis))
            width = 0.35
            axes[0].bar(x - width/2, gender_analysis['TotalPremium'], width, label='Total Premium', alpha=0.8)
            axes[0].bar(x + width/2, gender_analysis['TotalClaims'], width, label='Total Claims', alpha=0.8)
            axes[0].set_xlabel('Gender')
            axes[0].set_ylabel('Amount')
            axes[0].set_title('Premium and Claims by Gender')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(gender_analysis['Gender'])
            axes[0].legend()
            axes[0].grid(True, alpha=0.3, axis='y')
            
            # Loss Ratio by Gender
            colors = ['#FF6B6B' if lr > 0.5 else '#4ECDC4' for lr in gender_analysis['LossRatio']]
            axes[1].bar(gender_analysis['Gender'], gender_analysis['LossRatio'], color=colors, alpha=0.8, edgecolor='black')
            axes[1].axhline(y=0.5, color='r', linestyle='--', label='Break-even (0.5)')
            axes[1].set_xlabel('Gender')
            axes[1].set_ylabel('Loss Ratio')
            axes[1].set_title('Loss Ratio by Gender')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            if save:
                plt.savefig(self.reports_dir / 'creative_viz2_gender_risk_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualization 3: Top Vehicle Makes by Loss Ratio
        if 'Make' in self.df.columns:
            make_analysis = self.df.groupby('Make').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            make_analysis['LossRatio'] = make_analysis['TotalClaims'] / make_analysis['TotalPremium']
            make_analysis = make_analysis[make_analysis['PolicyID'] >= 10]  # Filter for makes with at least 10 policies
            top_makes = make_analysis.nlargest(15, 'PolicyID')  # Top 15 by policy count
            
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(top_makes['PolicyID'], top_makes['LossRatio'], 
                               s=top_makes['TotalPremium']/1000, alpha=0.6, 
                               c=top_makes['LossRatio'], cmap='RdYlGn_r', edgecolors='black')
            
            for idx, row in top_makes.iterrows():
                ax.annotate(row['Make'], (row['PolicyID'], row['LossRatio']), 
                          fontsize=8, ha='center')
            
            ax.set_xlabel('Number of Policies', fontsize=12)
            ax.set_ylabel('Loss Ratio', fontsize=12)
            ax.set_title('Vehicle Make Analysis: Policy Count vs Loss Ratio\n(Bubble size = Total Premium)', 
                        fontsize=14, fontweight='bold')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Break-even (0.5)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.colorbar(scatter, ax=ax, label='Loss Ratio')
            plt.tight_layout()
            if save:
                plt.savefig(self.reports_dir / 'creative_viz3_vehicle_make_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_eda_report(self):
        """
        Generate comprehensive EDA report with all analyses.
        """
        print("="*80)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Data Information
        print("\n1. DATA INFORMATION")
        print("-"*80)
        info = get_data_info(self.df)
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        
        # Data Quality
        print("\n2. DATA QUALITY ASSESSMENT")
        print("-"*80)
        quality = self.data_quality_assessment()
        print(f"Total Rows: {quality['total_rows']:,}")
        print(f"Total Columns: {quality['total_columns']}")
        print(f"Duplicate Rows: {quality['duplicate_rows']:,}")
        print("\nMissing Values:")
        for col, count in quality['missing_values'].items():
            if count > 0:
                pct = quality['missing_percentage'][col]
                print(f"  {col}: {count:,} ({pct:.2f}%)")
        
        # Descriptive Statistics
        print("\n3. DESCRIPTIVE STATISTICS")
        print("-"*80)
        desc_stats = self.data_summarization()
        print(desc_stats.head(10))
        
        # Loss Ratio Analysis
        print("\n4. LOSS RATIO ANALYSIS")
        print("-"*80)
        overall_loss = self.calculate_loss_ratio()
        print(f"Overall Loss Ratio: {overall_loss['LossRatio'].iloc[0]:.4f}")
        
        if 'Province' in self.df.columns:
            loss_by_province = self.calculate_loss_ratio(['Province'])
            print("\nLoss Ratio by Province:")
            print(loss_by_province.sort_values('LossRatio', ascending=False))
        
        if 'VehicleType' in self.df.columns:
            loss_by_vehicle = self.calculate_loss_ratio(['VehicleType'])
            print("\nLoss Ratio by Vehicle Type:")
            print(loss_by_vehicle.sort_values('LossRatio', ascending=False))
        
        if 'Gender' in self.df.columns:
            loss_by_gender = self.calculate_loss_ratio(['Gender'])
            print("\nLoss Ratio by Gender:")
            print(loss_by_gender.sort_values('LossRatio', ascending=False))
        
        # Outlier Detection
        print("\n5. OUTLIER DETECTION")
        print("-"*80)
        key_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CustomValueEstimate']
        available_cols = [col for col in key_cols if col in self.df.columns]
        outliers = self.detect_outliers(available_cols)
        for col, info in outliers.items():
            print(f"{col}: {info['count']:,} outliers ({info['percentage']:.2f}%)")
        
        print("\n" + "="*80)
        print("EDA COMPLETE - Visualizations saved to reports/ directory")
        print("="*80)


def main():
    """Main function to run EDA"""
    # Initialize EDA
    eda = InsuranceEDA()
    
    # Generate all analyses
    eda.generate_eda_report()
    
    # Create all visualizations
    print("\nGenerating visualizations...")
    eda.plot_univariate_distributions()
    eda.plot_bivariate_analysis()
    eda.plot_geographic_trends()
    eda.plot_outliers()
    eda.plot_temporal_trends()
    eda.create_creative_visualizations()
    
    print("\nAll EDA analyses and visualizations completed!")


if __name__ == "__main__":
    main()

