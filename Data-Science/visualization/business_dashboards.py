"""
Business Dashboard Visualizations
---------------------------------
Functions for creating business-focused dashboard visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

def create_sales_dashboard(sales_data, date_col='date', amount_col='amount', 
                          product_col='product', region_col='region', title=None):
    """Create comprehensive sales dashboard."""
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(sales_data[date_col]):
        sales_data[date_col] = pd.to_datetime(sales_data[date_col])
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create a complex subplot layout
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1, 1])
    
    # 1. Sales trend over time (main chart)
    ax1 = fig.add_subplot(gs[0, :])
    daily_sales = sales_data.groupby(sales_data[date_col].dt.date)[amount_col].sum()
    ax1.plot(daily_sales.index, daily_sales.values, linewidth=2, color='#2E86C1')
    ax1.fill_between(daily_sales.index, daily_sales.values, alpha=0.3, color='#2E86C1')
    
    # Add moving average
    if len(daily_sales) >= 7:
        ma_7 = daily_sales.rolling(window=7).mean()
        ax1.plot(ma_7.index, ma_7.values, '--', color='red', label='7-day MA')
    
    ax1.set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sales Amount')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Product performance
    ax2 = fig.add_subplot(gs[1, 0])
    if product_col in sales_data.columns:
        product_sales = sales_data.groupby(product_col)[amount_col].sum().sort_values(ascending=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(product_sales)))
        bars = ax2.barh(range(len(product_sales)), product_sales.values, color=colors)
        ax2.set_yticks(range(len(product_sales)))
        ax2.set_yticklabels(product_sales.index)
        ax2.set_title('Sales by Product', fontweight='bold')
        ax2.set_xlabel('Total Sales')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, product_sales.values)):
            ax2.text(value + max(product_sales.values) * 0.01, i, f'${value:,.0f}', 
                    va='center', fontsize=9)
    
    # 3. Regional performance
    ax3 = fig.add_subplot(gs[1, 1])
    if region_col in sales_data.columns:
        region_sales = sales_data.groupby(region_col)[amount_col].sum()
        wedges, texts, autotexts = ax3.pie(region_sales.values, labels=region_sales.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Sales by Region', fontweight='bold')
    
    # 4. Monthly comparison
    ax4 = fig.add_subplot(gs[1, 2])
    sales_data['month'] = sales_data[date_col].dt.to_period('M')
    monthly_sales = sales_data.groupby('month')[amount_col].sum()
    
    if len(monthly_sales) > 1:
        ax4.bar(range(len(monthly_sales)), monthly_sales.values, 
               color='skyblue', edgecolor='navy', alpha=0.7)
        ax4.set_xticks(range(len(monthly_sales)))
        ax4.set_xticklabels([str(m) for m in monthly_sales.index], rotation=45)
        ax4.set_title('Monthly Sales', fontweight='bold')
        ax4.set_ylabel('Sales Amount')
    
    # 5. Key metrics summary
    ax5 = fig.add_subplot(gs[1, 3])
    ax5.axis('off')
    
    # Calculate KPIs
    total_sales = sales_data[amount_col].sum()
    avg_daily_sales = daily_sales.mean()
    total_transactions = len(sales_data)
    avg_transaction_value = total_sales / total_transactions if total_transactions > 0 else 0
    
    # Growth calculation (if we have more than one day)
    if len(daily_sales) > 1:
        recent_sales = daily_sales.tail(7).mean()
        previous_sales = daily_sales.head(7).mean() if len(daily_sales) > 14 else daily_sales.iloc[0]
        growth_rate = (recent_sales - previous_sales) / previous_sales * 100 if previous_sales > 0 else 0
    else:
        growth_rate = 0
    
    kpi_text = f"""
    KEY METRICS
    
    Total Sales: ${total_sales:,.0f}
    
    Avg Daily Sales: ${avg_daily_sales:,.0f}
    
    Total Transactions: {total_transactions:,}
    
    Avg Transaction: ${avg_transaction_value:.2f}
    
    Growth Rate: {growth_rate:+.1f}%
    
    Best Day: ${daily_sales.max():,.0f}
    
    Date Range: {len(daily_sales)} days
    """
    
    ax5.text(0.05, 0.95, kpi_text, transform=ax5.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
             facecolor='lightblue', alpha=0.8))
    
    # 6. Sales distribution
    ax6 = fig.add_subplot(gs[2, :2])
    ax6.hist(sales_data[amount_col], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax6.axvline(sales_data[amount_col].mean(), color='red', linestyle='--', 
               label=f'Mean: ${sales_data[amount_col].mean():.2f}')
    ax6.axvline(sales_data[amount_col].median(), color='blue', linestyle='--', 
               label=f'Median: ${sales_data[amount_col].median():.2f}')
    ax6.set_title('Transaction Amount Distribution', fontweight='bold')
    ax6.set_xlabel('Transaction Amount')
    ax6.set_ylabel('Frequency')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Top customers/transactions
    ax7 = fig.add_subplot(gs[2, 2:])
    top_transactions = sales_data.nlargest(10, amount_col)
    
    if 'customer' in sales_data.columns:
        customer_sales = sales_data.groupby('customer')[amount_col].sum().nlargest(5)
        ax7.bar(range(len(customer_sales)), customer_sales.values, color='orange')
        ax7.set_xticks(range(len(customer_sales)))
        ax7.set_xticklabels(customer_sales.index, rotation=45)
        ax7.set_title('Top 5 Customers', fontweight='bold')
    else:
        ax7.bar(range(len(top_transactions)), top_transactions[amount_col].values, color='orange')
        ax7.set_title('Top 10 Transactions', fontweight='bold')
        ax7.set_xlabel('Transaction Rank')
    
    ax7.set_ylabel('Amount')
    
    plt.suptitle(title or 'Sales Performance Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_kpi_dashboard(metrics_dict, targets_dict=None, title=None):
    """Create KPI dashboard with gauges and indicators."""
    n_metrics = len(metrics_dict)
    cols = min(4, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (metric_name, current_value) in enumerate(metrics_dict.items()):
        ax = axes[i] if i < len(axes) else None
        if ax is None:
            break
            
        # Get target if available
        target_value = targets_dict.get(metric_name) if targets_dict else None
        
        # Create gauge chart
        if target_value:
            # Calculate percentage of target achieved
            percentage = (current_value / target_value) * 100
            
            # Color based on performance
            if percentage >= 100:
                color = 'green'
            elif percentage >= 75:
                color = 'yellow'
            else:
                color = 'red'
            
            # Create semicircle gauge
            theta = np.linspace(0, np.pi, 100)
            r = 1
            
            # Background arc
            ax.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=8)
            
            # Progress arc
            progress_theta = np.linspace(0, np.pi * min(percentage/100, 1), 100)
            ax.plot(r * np.cos(progress_theta), r * np.sin(progress_theta), 
                   color, linewidth=8)
            
            # Add needle
            needle_angle = np.pi * (1 - min(percentage/100, 1))
            ax.plot([0, 0.8 * np.cos(needle_angle)], [0, 0.8 * np.sin(needle_angle)], 
                   'black', linewidth=3)
            
            # Add text
            ax.text(0, -0.3, f'{current_value:,.0f}', ha='center', va='center', 
                   fontsize=14, fontweight='bold')
            ax.text(0, -0.5, f'Target: {target_value:,.0f}', ha='center', va='center', 
                   fontsize=10)
            ax.text(0, -0.7, f'{percentage:.1f}%', ha='center', va='center', 
                   fontsize=12, color=color, fontweight='bold')
            
        else:
            # Simple value display
            ax.text(0.5, 0.5, f'{current_value:,.1f}', ha='center', va='center', 
                   fontsize=20, fontweight='bold', transform=ax.transAxes)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(metric_name.replace('_', ' ').title(), fontweight='bold')
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title or 'KPI Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def create_financial_dashboard(financial_data, date_col='date', title=None):
    """Create financial performance dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1, 1])
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(financial_data[date_col]):
        financial_data[date_col] = pd.to_datetime(financial_data[date_col])
    
    # 1. Main financial trend chart
    ax1 = fig.add_subplot(gs[0, :])
    
    # Plot multiple financial metrics if available
    metrics_to_plot = ['revenue', 'profit', 'expenses', 'net_income']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    
    for i, metric in enumerate(metrics_to_plot):
        if metric in financial_data.columns:
            daily_data = financial_data.groupby(financial_data[date_col].dt.date)[metric].sum()
            ax1.plot(daily_data.index, daily_data.values, 
                    label=metric.title(), color=colors[i], linewidth=2)
    
    ax1.set_title('Financial Performance Trend', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Amount ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Profit margin analysis
    ax2 = fig.add_subplot(gs[1, 0])
    if 'revenue' in financial_data.columns and 'profit' in financial_data.columns:
        financial_data['profit_margin'] = (financial_data['profit'] / financial_data['revenue']) * 100
        monthly_margin = financial_data.groupby(financial_data[date_col].dt.to_period('M'))['profit_margin'].mean()
        
        bars = ax2.bar(range(len(monthly_margin)), monthly_margin.values, 
                      color=['green' if x > 0 else 'red' for x in monthly_margin.values])
        ax2.set_xticks(range(len(monthly_margin)))
        ax2.set_xticklabels([str(m) for m in monthly_margin.index], rotation=45)
        ax2.set_title('Monthly Profit Margin %', fontweight='bold')
        ax2.set_ylabel('Margin %')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # 3. Revenue breakdown (if categories available)
    ax3 = fig.add_subplot(gs[1, 1])
    if 'category' in financial_data.columns and 'revenue' in financial_data.columns:
        category_revenue = financial_data.groupby('category')['revenue'].sum()
        ax3.pie(category_revenue.values, labels=category_revenue.index, autopct='%1.1f%%')
        ax3.set_title('Revenue by Category', fontweight='bold')
    
    # 4. Cash flow analysis
    ax4 = fig.add_subplot(gs[1, 2])
    if 'cash_flow' in financial_data.columns:
        daily_cashflow = financial_data.groupby(financial_data[date_col].dt.date)['cash_flow'].sum()
        cumulative_cashflow = daily_cashflow.cumsum()
        
        ax4.plot(cumulative_cashflow.index, cumulative_cashflow.values, color='purple', linewidth=2)
        ax4.fill_between(cumulative_cashflow.index, cumulative_cashflow.values, 
                        alpha=0.3, color='purple')
        ax4.set_title('Cumulative Cash Flow', fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # 5. Financial ratios summary
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate key financial metrics
    if 'revenue' in financial_data.columns:
        total_revenue = financial_data['revenue'].sum()
        avg_revenue = financial_data['revenue'].mean()
    else:
        total_revenue = avg_revenue = 0
    
    if 'expenses' in financial_data.columns:
        total_expenses = financial_data['expenses'].sum()
    else:
        total_expenses = 0
    
    net_profit = total_revenue - total_expenses
    profit_margin = (net_profit / total_revenue * 100) if total_revenue > 0 else 0
    
    # Create metrics boxes
    metrics = [
        ('Total Revenue', f'${total_revenue:,.0f}'),
        ('Total Expenses', f'${total_expenses:,.0f}'),
        ('Net Profit', f'${net_profit:,.0f}'),
        ('Profit Margin', f'{profit_margin:.1f}%'),
        ('Avg Daily Revenue', f'${avg_revenue:,.0f}')
    ]
    
    box_width = 0.18
    for i, (metric_name, value) in enumerate(metrics):
        x_pos = i * 0.2 + 0.05
        
        # Color based on positive/negative for profit metrics
        if 'Profit' in metric_name or 'Margin' in metric_name:
            box_color = 'lightgreen' if 'profit' in value.lower() and float(value.replace('$', '').replace(',', '').replace('%', '')) > 0 else 'lightcoral'
        else:
            box_color = 'lightblue'
        
        # Create metric box
        rect = Rectangle((x_pos, 0.3), box_width, 0.4, 
                        facecolor=box_color, edgecolor='black', alpha=0.7)
        ax5.add_patch(rect)
        
        # Add text
        ax5.text(x_pos + box_width/2, 0.6, value, ha='center', va='center', 
                fontsize=14, fontweight='bold')
        ax5.text(x_pos + box_width/2, 0.4, metric_name, ha='center', va='center', 
                fontsize=10, wrap=True)
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    
    plt.suptitle(title or 'Financial Performance Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig