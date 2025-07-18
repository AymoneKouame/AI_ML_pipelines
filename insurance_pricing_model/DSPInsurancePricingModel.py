"""
DSP Insurance Premium Pricing Model
A comprehensive statistical modeling project for Delivery Service Partner insurance pricing

This project demonstrates:
1. Advanced statistical modeling techniques
2. Economic analysis and pricing optimization
3. Risk assessment and portfolio management
4. Market trend analysis and competitive intelligence
5. Performance monitoring and business insights

Author: [Aymone Kouame]
Date: July 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class DSPInsurancePricingModel:
    """
    Comprehensive insurance pricing model for Delivery Service Partners
    
    This model captures:
    - Geographic risk factors
    - Operational performance metrics
    - Market competition dynamics
    - Economic indicators
    - Portfolio optimization strategies
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        

    def exploratory_analysis(self, df):
        """
        Comprehensive exploratory data analysis
        """
        print("=== DSP Insurance Pricing Model - Exploratory Analysis ===\n")
        
        # 1. Basic statistics
        print("~~~~~~~ Dataset Overview:")
        display(df.info())
        print(f"Shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print("\nTarget Variable (Annual Premium) Statistics:")
        print(df['annual_premium'].describe())

        ## Premium distribution
        df.hist(column='annual_premium')
        plt.title('Distribution of Annual Insurance Premium')
        plt.suptitle('')
        plt.show()
            
        # 2. Correlation analysis
        print("\n=== Correlation Analysis of Numeric Columns ===")
        num_cols_df = df.select_dtypes(include='number')
        correlation_matrix = num_cols_df.corr()

        ### Print the correlation matrix
        print("~~~~~~~ Correlation Matrix Table (%)")
        display(correlation_matrix.apply(lambda x: round(x*100)))

        ### Plot the correlation matrix as a heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".0%")
        plt.title('Correlation Matrix Heatmap (%)')
        plt.show()
        
        
        # premium_corr = correlation_matrix['annual_premium'].sort_values(ascending=False)
        # print("\nTop factors correlated with premium:")
        # for factor, corr in premium_corr.items():
        #     if factor != 'annual_premium':
        #         print(f"{factor}: {corr:.3f}")
        # Top correlations with premium (CORR >50%)
        premium_corr_50plus = correlation_matrix[['annual_premium']]
        premium_corr_50plus = premium_corr_50plus[premium_corr_50plus.annual_premium >=50]

        print("~~~~~~~ Scatter Plot of factors correlated with premium (>=50%):")
        for col in [c for c in premium_corr_50plus.columns if c != 'annual_premium']:
            print(col)
            df.plot('annual_premium', col, kind = 'scatter')
            plt.title(f'Annual Premium Vs {col.title()}')
            plt.suptitle('')
            plt.ylabel("Annual Premium ($)")
            plt.xlabel(f"{col.replace('_',' ').title()}")
            plt.show()
   
        print("\n=== Distribution of Categorical or String Columns ===")
        cat_cols_df = df.select_dtypes(include=['object','category','string'])
        for c in cat_cols_df.columns:
            print('~~~~ '+c)
            display(df[['annual_premium', c]].groupby(c).mean().rename(columns = {'annual_premium':'avg_annual_premium'}))
            # Premium by col
            df.boxplot(column='annual_premium', by=c)
            plt.title(f'Premium Distribution by {c.title()}')
            plt.suptitle('')
            plt.ylabel("Annual Premium ($)")
            plt.xlabel(f"{c.replace('_',' ').title()}")
            plt.show()
            
        
        # Risk segmentation analysis
        print("\n=== Risk Segmentation Analysis ===")
        
        # Create risk buckets
        df['risk_segment'] = pd.cut(df['annual_premium'], 
                                   bins=5, 
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        risk_analysis = df.groupby('risk_segment').agg({
            'fleet_size': 'mean',
            'safety_incidents_per_1000': 'mean',
            'delivery_success_rate': 'mean',
            'historical_claims_ratio': 'mean',
            'annual_premium': 'mean'
        }).round(2)
        
        print("Risk Segment Characteristics:")
        display(risk_analysis)
        
        return df
    
    def advanced_statistical_modeling(self, df):
        """
        Build and compare multiple statistical models
        """
        print("\n=== Advanced Statistical Modeling ===\n")
        
        # Prepare features
        X = df.copy()
        y = X.pop('annual_premium')
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col])
        
        # Remove risk_segment if it exists (derived from target)
        if 'risk_segment' in X.columns:
            X = X.drop('risk_segment', axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        print("Model Performance Comparison:")
        print("-" * 70)
        print(f"{'Model':<20} {'R² Score':<12} {'RMSE':<12} {'MAE':<12}")
        print("-" * 70)
        
        for name, model in models.items():
            # Train model
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'predictions': y_pred
            }
            
            print(f"{name:<20} {r2:<12.4f} {rmse:<12.0f} {mae:<12.0f}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        best_model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"R² Score: {results[best_model_name]['r2']:.4f}")
        
        # Feature importance analysis
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
            
            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.xlabel('Importance Score')
            plt.tight_layout()
            plt.show()
        
        # Store best model
        self.models['best_model'] = best_model
        self.models['best_model_name'] = best_model_name
        
        return results, X_test, y_test
    
    def pricing_optimization_analysis(self, df):
        """
        Advanced pricing optimization and portfolio analysis
        """
        print("\n=== Pricing Optimization Analysis ===\n")
        
        # Portfolio risk analysis
        portfolio_stats = {
            'total_policies': len(df),
            'total_premium_volume': df['annual_premium'].sum(),
            'avg_premium': df['annual_premium'].mean(),
            'premium_std': df['annual_premium'].std(),
            'coefficient_of_variation': df['annual_premium'].std() / df['annual_premium'].mean()
        }
        
        print("Portfolio Overview:")
        for key, value in portfolio_stats.items():
            if 'total' in key or 'avg' in key:
                print(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        
        # Risk-adjusted pricing analysis
        print("\n=== Risk-Adjusted Pricing Analysis ===")
        
        # Calculate loss ratios (simplified)
        df['expected_loss'] = df['historical_claims_ratio'] * df['avg_claim_amount']
        df['loss_ratio'] = df['expected_loss'] / df['annual_premium']
        df['profit_margin'] = (df['annual_premium'] - df['expected_loss']) / df['annual_premium']
        
        # Portfolio segmentation
        df['profitability_tier'] = pd.cut(df['profit_margin'], 
                                         bins=5, 
                                         labels=['Loss Making', 'Low Profit', 'Medium Profit', 'High Profit', 'Premium'])
        
        profitability_analysis = df.groupby('profitability_tier').agg({
            'annual_premium': ['count', 'mean', 'sum'],
            'loss_ratio': 'mean',
            'profit_margin': 'mean',
            'fleet_size': 'mean'
        }).round(3)
        
        print("Profitability Tier Analysis:")
        print(profitability_analysis)
        
        # Competitive analysis simulation
        print("\n=== Market Competition Analysis ===")
        
        # Simulate competitive pricing scenarios
        competition_scenarios = {
            'current_pricing': df['annual_premium'].mean(),
            'aggressive_pricing': df['annual_premium'].mean() * 0.9,
            'premium_pricing': df['annual_premium'].mean() * 1.1,
            'market_leader': df['annual_premium'].mean() * 1.05
        }
        
        # Estimate market share impact (simplified model)
        for scenario, price in competition_scenarios.items():
            price_elasticity = -1.5  # Assumed elasticity
            price_change = (price - df['annual_premium'].mean()) / df['annual_premium'].mean()
            market_share_change = price_elasticity * price_change
            estimated_market_share = 0.15 * (1 + market_share_change)  # Base 15% market share
            
            print(f"{scenario.replace('_', ' ').title()}: ${price:,.0f} "
                  f"(Est. Market Share: {estimated_market_share:.1%})")
        
        # Pricing recommendations
        print("\n=== Pricing Recommendations ===")
        
        # High-risk segments that need price increases
        high_risk = df[df['loss_ratio'] > 0.8]
        print(f"High-risk policies (loss ratio > 80%): {len(high_risk)} policies")
        print(f"Recommended average price increase: {((1/high_risk['loss_ratio'].mean()) - 1) * 100:.1f}%")
        
        # Profitable segments for growth
        profitable = df[df['profit_margin'] > 0.3]
        print(f"Highly profitable policies (>30% margin): {len(profitable)} policies")
        print(f"Potential for competitive pricing to gain market share")
        
        return df
    
    def generate_business_insights(self, df):
        """
        Generate comprehensive business insights and recommendations
        """
        print("\n=== Business Insights & Strategic Recommendations ===\n")
        
        # Key insights
        insights = []
        
        # Fleet size impact
        fleet_correlation = df['fleet_size'].corr(df['annual_premium'])
        insights.append(f"Fleet size has {fleet_correlation:.3f} correlation with premium - "
                       f"{'strong' if abs(fleet_correlation) > 0.5 else 'moderate'} relationship")
        
        # Safety performance
        safety_correlation = df['safety_incidents_per_1000'].corr(df['annual_premium'])
        insights.append(f"Safety incidents show {safety_correlation:.3f} correlation with premium")
        
        # Regional variations
        regional_std = df.groupby('region')['annual_premium'].mean().std()
        insights.append(f"Regional premium variation: ${regional_std:,.0f} standard deviation")
        
        # Market opportunity
        low_competition = df[df['competitor_density'] < 2]
        insights.append(f"Low competition markets: {len(low_competition)} policies "
                       f"({len(low_competition)/len(df)*100:.1f}% of portfolio)")
        
        print("Key Business Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        # Strategic recommendations
        print("\nStrategic Recommendations:")
        
        recommendations = [
            "Implement dynamic pricing based on real-time safety performance metrics",
            "Expand in low-competition markets with competitive pricing strategy",
            "Develop safety incentive programs to reduce claims and improve profitability",
            "Create regional pricing models to capture local market dynamics",
            "Invest in telematics and IoT for better risk assessment",
            "Build predictive models for early identification of high-risk DSPs"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
        
        # ROI projections
        print("\nProjected Impact of Recommendations:")
        
        # Safety program ROI
        safety_improvement = 0.2  # 20% reduction in incidents
        current_avg_loss = df['expected_loss'].mean()
        safety_savings = current_avg_loss * safety_improvement
        print(f"Safety program potential savings: ${safety_savings:,.0f} per policy annually")
        
        # Market expansion ROI
        expansion_policies = 1000  # New policies in low-competition markets
        avg_premium = df['annual_premium'].mean()
        avg_margin = df['profit_margin'].mean()
        expansion_revenue = expansion_policies * avg_premium * avg_margin
        print(f"Market expansion potential profit: ${expansion_revenue:,.0f} annually")
        
        return insights, recommendations

def run_insurance_pricing_model(df):
    """
    Main execution function - demonstrates the complete modeling pipeline
    """
    print("DSP Insurance Premium Pricing Model")
    print("=" * 50)
    
    # Initialize model
    model = DSPInsurancePricingModel()
        
    # Exploratory analysis
    df = model.exploratory_analysis(df)
    
    # Statistical modeling
    results, X_test, y_test = model.advanced_statistical_modeling(df)
    
    # Pricing optimization
    df = model.pricing_optimization_analysis(df)
    
    # Business insights
    insights, recommendations = model.generate_business_insights(df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("PROJECT SUMMARY")
    print("=" * 60)
    print(f"✓ Analyzed {len(df):,} DSP insurance policies")
    print(f"✓ Built {len(results)} predictive models")
    print(f"✓ Best model: {model.models['best_model_name']}")
    print(f"✓ Generated {len(insights)} key business insights")
    print(f"✓ Provided {len(recommendations)} strategic recommendations")
    print("\nThis comprehensive analysis demonstrates:")
    print("- Advanced statistical modeling techniques")
    print("- Economic analysis and pricing optimization")
    print("- Risk assessment and portfolio management")
    print("- Market intelligence and competitive analysis")
    print("- Business strategy and decision support")
    
    return model, df, results