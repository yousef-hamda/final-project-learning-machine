import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class CryptoDataCollector:
    """
    A comprehensive class for collecting and preparing cryptocurrency data
    This class handles data collection from CoinGecko API and creates features for ML models
    """
    
    def __init__(self):
        # Base URL for CoinGecko API - free and reliable crypto data source
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # List of cryptocurrencies to analyze - focusing on major coins
        self.coins = ['bitcoin', 'ethereum', 'binancecoin']
        
        # Base currency for price comparison
        self.vs_currency = 'usd'
        
    def get_historical_data(self, coin_id, days=365):
        """
        Fetch historical data for a specific cryptocurrency
        
        Parameters:
        -----------
        coin_id : str
            The ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        days : int
            Number of days of historical data to fetch (default: 365)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing historical price, volume, and market cap data
        """
        # Construct the API endpoint URL
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        
        # Parameters for the API request
        params = {
            'vs_currency': self.vs_currency,  # Price in USD
            'days': days,                     # Historical period
            'interval': 'daily'               # Daily data points
        }
        
        try:
            # Make HTTP request to CoinGecko API
            print(f"Fetching data for {coin_id}...")
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise exception for bad status codes
            data = response.json()
            
            # Extract price, volume, and market cap data from API response
            prices = data['prices']           # List of [timestamp, price] pairs
            volumes = data['total_volumes']   # List of [timestamp, volume] pairs
            market_caps = data['market_caps'] # List of [timestamp, market_cap] pairs
            
            # Create DataFrame from the extracted data
            df = pd.DataFrame({
                'timestamp': [item[0] for item in prices],      # Extract timestamps
                'price': [item[1] for item in prices],          # Extract prices
                'volume': [item[1] for item in volumes],        # Extract volumes
                'market_cap': [item[1] for item in market_caps] # Extract market caps
            })
            
            # Convert timestamp from milliseconds to readable date format
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add coin identifier for multi-coin analysis
            df['coin'] = coin_id
            
            print(f"Successfully fetched {len(df)} records for {coin_id}")
            return df
            
        except requests.exceptions.RequestException as e:
            # Handle API request errors gracefully
            print(f"Error fetching data for {coin_id}: {e}")
            return None
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators used in cryptocurrency trading
        These indicators help identify trends and potential buy/sell signals
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added technical indicator columns
        """
        print("Calculating technical indicators...")
        
        # Moving Averages - smooth out price fluctuations to identify trends
        df['ma_7'] = df['price'].rolling(window=7).mean()    # 7-day moving average
        df['ma_14'] = df['price'].rolling(window=14).mean()  # 14-day moving average  
        df['ma_30'] = df['price'].rolling(window=30).mean()  # 30-day moving average
        
        # RSI (Relative Strength Index) - momentum oscillator (0-100)
        # Values above 70 indicate overbought, below 30 indicate oversold
        delta = df['price'].diff()  # Daily price changes
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()  # Average gains
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean() # Average losses
        rs = gain / loss  # Relative strength
        df['rsi'] = 100 - (100 / (1 + rs))  # RSI formula
        
        # MACD (Moving Average Convergence Divergence) - trend following indicator
        exp1 = df['price'].ewm(span=12).mean()  # 12-period EMA
        exp2 = df['price'].ewm(span=26).mean()  # 26-period EMA
        df['macd'] = exp1 - exp2                # MACD line
        df['macd_signal'] = df['macd'].ewm(span=9).mean()  # Signal line
        df['macd_histogram'] = df['macd'] - df['macd_signal']  # Histogram
        
        # Bollinger Bands - volatility bands around moving average
        df['bb_middle'] = df['price'].rolling(window=20).mean()  # Middle band (20-day MA)
        bb_std = df['price'].rolling(window=20).std()            # Standard deviation
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)          # Upper band
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)          # Lower band
        
        print("Technical indicators calculated successfully")
        return df
    
    def calculate_price_features(self, df):
        """
        Calculate price-based features for machine learning models
        These features capture price momentum and volatility patterns
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing price and technical indicator data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional price-based feature columns
        """
        print("Calculating price features...")
        
        # Percentage changes - capture momentum at different time horizons
        df['price_change_1d'] = df['price'].pct_change(1)   # Daily return
        df['price_change_7d'] = df['price'].pct_change(7)   # Weekly return
        df['price_change_30d'] = df['price'].pct_change(30) # Monthly return
        
        # Volatility measures - capture market uncertainty
        df['volatility_7d'] = df['price_change_1d'].rolling(window=7).std()   # 7-day volatility
        df['volatility_30d'] = df['price_change_1d'].rolling(window=30).std() # 30-day volatility
        
        # Price ratios - relative position compared to moving averages
        df['price_to_ma7_ratio'] = df['price'] / df['ma_7']   # Price vs 7-day MA
        df['price_to_ma30_ratio'] = df['price'] / df['ma_30'] # Price vs 30-day MA
        
        # Volume indicators - trading activity analysis
        df['volume_change_1d'] = df['volume'].pct_change(1)              # Daily volume change
        df['volume_ma_7'] = df['volume'].rolling(window=7).mean()        # 7-day volume average
        df['volume_ratio'] = df['volume'] / df['volume_ma_7']            # Volume vs average
        
        print("Price features calculated successfully")
        return df
    
    def add_time_features(self, df):
        """
        Add time-based features that might affect cryptocurrency prices
        Market behavior often shows patterns based on time (day of week, month, etc.)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing date information
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional time-based feature columns
        """
        print("Adding time features...")
        
        # Day of week (0=Monday, 6=Sunday) - crypto markets show weekly patterns
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Month (1-12) - seasonal effects in crypto markets
        df['month'] = df['date'].dt.month
        
        # Quarter (1-4) - quarterly market cycles
        df['quarter'] = df['date'].dt.quarter
        
        # Weekend indicator - crypto markets trade 24/7 but show weekend patterns
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print("Time features added successfully")
        return df
    
    def create_target_variable(self, df, prediction_days=1):
        """
        Create target variables for machine learning prediction
        We predict if price will go up or down in the next period
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing all features
        prediction_days : int
            Number of days ahead to predict (default: 1 day)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with target variables for classification and regression
        """
        print(f"Creating target variable for {prediction_days}-day prediction...")
        
        # Future price - shift price data backwards to get future values
        df['future_price'] = df['price'].shift(-prediction_days)
        
        # Binary classification target: 1 if price goes up, 0 if price goes down
        df['target'] = (df['future_price'] > df['price']).astype(int)
        
        # Regression target: percentage change in price
        df['target_return'] = (df['future_price'] / df['price'] - 1) * 100
        
        # Multi-class classification target based on magnitude of change
        conditions = [
            df['target_return'] <= -2,  # Strong decrease (class 0)
            (df['target_return'] > -2) & (df['target_return'] <= 0),  # Mild decrease (class 1)
            (df['target_return'] > 0) & (df['target_return'] <= 2),   # Mild increase (class 2)
            df['target_return'] > 2     # Strong increase (class 3)
        ]
        choices = [0, 1, 2, 3]
        df['target_class'] = np.select(conditions, choices, default=1)
        
        print("Target variables created successfully")
        return df
    
    def collect_all_data(self):
        """
        Main method to collect and process data for all cryptocurrencies
        This orchestrates the entire data collection and feature engineering pipeline
        
        Returns:
        --------
        pandas.DataFrame
            Complete dataset ready for machine learning, or None if collection fails
        """
        print("="*60)
        print("STARTING CRYPTOCURRENCY DATA COLLECTION")
        print("="*60)
        
        all_data = []  # List to store data from all coins
        
        # Process each cryptocurrency
        for coin in self.coins:
            print(f"\nProcessing {coin.upper()}...")
            
            # Step 1: Fetch historical data from API
            df = self.get_historical_data(coin, days=365)
            
            if df is not None:
                # Step 2: Calculate technical indicators
                df = self.calculate_technical_indicators(df)
                
                # Step 3: Calculate price-based features
                df = self.calculate_price_features(df)
                
                # Step 4: Add time-based features
                df = self.add_time_features(df)
                
                # Step 5: Create target variables
                df = self.create_target_variable(df)
                
                # Add processed data to collection
                all_data.append(df)
                print(f"✅ {coin.upper()} processed successfully!")
                
                # Rate limiting - avoid overwhelming the API
                time.sleep(1)
            else:
                print(f"❌ Failed to process {coin.upper()}")
        
        # Combine data from all cryptocurrencies
        if all_data:
            print(f"\nCombining data from {len(all_data)} cryptocurrencies...")
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✅ Combined dataset created with {len(combined_df)} total records")
            return combined_df
        else:
            print("❌ No data collected - check your internet connection and API availability")
            return None
    
    def clean_and_prepare_data(self, df):
        """
        Clean and prepare the final dataset for machine learning
        Remove missing values and select relevant features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Raw combined dataset
            
        Returns:
        --------
        tuple
            (X, y_binary, y_multi, clean_df) - Features, targets, and clean dataset
        """
        print("\n" + "="*60)
        print("CLEANING AND PREPARING DATA FOR MACHINE LEARNING")
        print("="*60)
        
        # Remove rows with missing values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        print(f"Removed {initial_rows - final_rows} rows with missing values")
        
        # Select relevant features for machine learning
        feature_columns = [
            # Price and market data
            'price', 'volume', 'market_cap',
            
            # Moving averages
            'ma_7', 'ma_14', 'ma_30',
            
            # Technical indicators
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            
            # Bollinger Bands
            'bb_upper', 'bb_lower', 'bb_middle',
            
            # Price features
            'price_change_1d', 'price_change_7d', 'price_change_30d',
            'volatility_7d', 'volatility_30d',
            'price_to_ma7_ratio', 'price_to_ma30_ratio',
            
            # Volume features
            'volume_change_1d', 'volume_ratio',
            
            # Time features
            'day_of_week', 'month', 'quarter', 'is_weekend'
        ]
        
        # Keep only features that exist in the dataset
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"Selected {len(available_features)} features for modeling")
        
        # Create feature matrix and target vectors
        X = df[available_features].copy()                    # Feature matrix
        y_binary = df['target'].copy()                       # Binary classification target
        y_multi = df['target_class'].copy()                  # Multi-class classification target
        
        print(f"Final dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Binary target distribution: {y_binary.value_counts().to_dict()}")
        print(f"Multi-class target distribution: {y_multi.value_counts().to_dict()}")
        
        return X, y_binary, y_multi, df

# Example usage and testing
if __name__ == "__main__":
    # Create data collector instance
    collector = CryptoDataCollector()
    
    # Collect all cryptocurrency data
    print("Starting cryptocurrency data collection...")
    data = collector.collect_all_data()
    
    if data is not None:
        print(f"\n✅ Successfully collected {len(data)} records")
        
        # Clean and prepare data for machine learning
        X, y_binary, y_multi, clean_data = collector.clean_and_prepare_data(data)
        
        print(f"\n📊 FINAL RESULTS:")
        print(f"Features shape: {X.shape}")
        print(f"Binary targets: {len(y_binary)} samples")
        print(f"Multi-class targets: {len(y_multi)} samples")
        
        # Save the dataset
        clean_data.to_csv('../data/crypto_data.csv', index=False)
        print(f"\n💾 Data saved to crypto_data.csv")
        
    else:
        print("\n❌ Data collection failed")