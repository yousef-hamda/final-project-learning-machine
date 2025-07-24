"""
Quick test script for debugging my ML algorithms
I made this because I kept breaking things and wanted to test quickly

Run this with: python test_my_code.py
"""

import numpy as np
import pandas as pd
import sys
import time

# Test if my imports work
print("🔧 Testing imports...")
try:
    from decision_tree import DecisionTree
    from random_forest import RandomForest  
    from adaboost import AdaBoost
    print("✅ All my custom classes imported successfully!")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_basic_functionality():
    """
    Test each algorithm with tiny dataset
    This helped me debug when things weren't working
    """
    print("\n📊 Creating tiny test dataset...")
    
    # Super simple test data - just 100 samples
    np.random.seed(42)
    X_test = np.random.rand(100, 5)  # 5 features
    y_test = np.random.choice([0, 1], 100)  # binary target
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Target distribution: {np.bincount(y_test)}")
    
    # Split for testing
    X_train = X_test[:80]
    y_train = y_test[:80] 
    X_val = X_test[80:]
    y_val = y_test[80:]
    
    models_to_test = {
        'Decision Tree': DecisionTree(max_depth=3, min_samples_split=5),
        'Random Forest': RandomForest(n_estimators=5, max_depth=3),  # small for speed
        'AdaBoost': AdaBoost(num_estimators=5)  # also small
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        print(f"\n🧪 Testing {name}...")
        
        try:
            # Train the model
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(X_val)
            
            # Calculate basic accuracy
            accuracy = np.mean(y_pred == y_val)
            
            results[name] = {
                'accuracy': accuracy,
                'train_time': train_time,
                'predictions_shape': y_pred.shape,
                'status': 'SUCCESS'
            }
            
            print(f"   ✅ {name}: accuracy={accuracy:.3f}, time={train_time:.3f}s")
            
        except Exception as e:
            results[name] = {
                'status': 'FAILED',
                'error': str(e)
            }
            print(f"   ❌ {name} failed: {e}")
    
    return results

def test_data_shapes():
    """
    Check if my data processing works correctly
    I had so many shape errors while coding this...
    """
    print("\n📐 Testing data shapes and processing...")
    
    # Create some fake crypto-like data
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    fake_data = pd.DataFrame({
        'date': dates,
        'price': 100 + np.cumsum(np.random.normal(0, 1, 50)),
        'volume': np.random.lognormal(10, 0.5, 50)
    })
    
    # Add some technical indicators like in my main code
    fake_data['ma_7'] = fake_data['price'].rolling(7).mean()
    fake_data['price_change'] = fake_data['price'].pct_change()
    fake_data['target'] = (fake_data['price'].shift(-1) > fake_data['price']).astype(int)
    
    # Clean data
    clean_data = fake_data.dropna()
    
    print(f"Original data shape: {fake_data.shape}")
    print(f"Clean data shape: {clean_data.shape}")
    print(f"Target distribution: {clean_data['target'].value_counts().to_dict()}")
    
    # Test feature selection
    feature_cols = ['price', 'volume', 'ma_7', 'price_change']
    X = clean_data[feature_cols]
    y = clean_data['target']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Any NaN values? {X.isnull().sum().sum()}")
    
    return X, y

def test_edge_cases():
    """
    Test some edge cases that broke my code before
    These were painful to debug...
    """
    print("\n🚨 Testing edge cases...")
    
    # Test 1: All same class
    print("Test 1: All samples same class")
    X_same = np.random.rand(20, 3)
    y_same = np.ones(20)  # all class 1
    
    try:
        dt = DecisionTree(max_depth=3)
        dt.fit(X_same, y_same)
        pred = dt.predict(X_same[:5])
        print(f"   ✅ Same class test passed: predictions = {pred}")
    except Exception as e:
        print(f"   ❌ Same class test failed: {e}")
    
    # Test 2: Very small dataset
    print("Test 2: Tiny dataset (10 samples)")
    X_tiny = np.random.rand(10, 2)
    y_tiny = np.random.choice([0, 1], 10)
    
    try:
        rf = RandomForest(n_estimators=3, max_depth=2)
        rf.fit(X_tiny, y_tiny)
        pred = rf.predict(X_tiny[:3])
        print(f"   ✅ Tiny dataset test passed: predictions = {pred}")
    except Exception as e:
        print(f"   ❌ Tiny dataset test failed: {e}")
    
    # Test 3: Single feature
    print("Test 3: Single feature only")
    X_single = np.random.rand(50, 1)  # only 1 feature
    y_single = np.random.choice([0, 1], 50)
    
    try:
        ada = AdaBoost(num_estimators=3)
        ada.fit(X_single, y_single)
        pred = ada.predict(X_single[:5])
        print(f"   ✅ Single feature test passed: predictions = {pred}")
    except Exception as e:
        print(f"   ❌ Single feature test failed: {e}")

def check_my_algorithms():
    """
    Sanity check - make sure my algorithms give reasonable results
    I spent way too much time debugging silly mistakes
    """
    print("\n🔍 Sanity checking my algorithms...")
    
    # Create dataset where pattern should be learnable
    np.random.seed(123)
    n_samples = 200
    
    # Feature 1: if > 0.5, class = 1, else class = 0 (should be easy to learn)
    X_simple = np.random.rand(n_samples, 3)
    y_simple = (X_simple[:, 0] > 0.5).astype(int)  # based on first feature
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X_simple[:split_idx], X_simple[split_idx:]
    y_train, y_test = y_simple[:split_idx], y_simple[split_idx:]
    
    print(f"Simple pattern: if feature_0 > 0.5 then class=1")
    print(f"Training accuracy should be high if algorithms work correctly")
    
    models = {
        'Decision Tree': DecisionTree(max_depth=5),
        'Random Forest': RandomForest(n_estimators=10, max_depth=5),
        'AdaBoost': AdaBoost(num_estimators=10)
    }
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            
            # Test on training data (should be high accuracy)
            train_pred = model.predict(X_train)
            train_acc = np.mean(train_pred == y_train)
            
            # Test on test data
            test_pred = model.predict(X_test)
            test_acc = np.mean(test_pred == y_test)
            
            print(f"   {name}: train_acc={train_acc:.3f}, test_acc={test_acc:.3f}")
            
            # Sanity check: should be much better than random (0.5)
            if train_acc > 0.7:
                print(f"   ✅ {name} learned the pattern correctly")
            else:
                print(f"   ⚠️ {name} might have issues (accuracy too low)")
                
        except Exception as e:
            print(f"   ❌ {name} crashed: {e}")

def main():
    """
    Run all my tests
    I usually run this before running the full project to catch bugs early
    """
    print("🧪 QUICK ALGORITHM TESTING")
    print("=" * 50)
    print("This is my debugging script - helps me catch issues fast")
    print("If everything passes, the main project should work fine")
    print("=" * 50)
    
    # Run all tests
    try:
        print("\n🔸 Test 1: Basic Functionality")
        basic_results = test_basic_functionality()
        
        print("\n🔸 Test 2: Data Shape Processing")
        X_test, y_test = test_data_shapes()
        
        print("\n🔸 Test 3: Edge Cases")
        test_edge_cases()
        
        print("\n🔸 Test 4: Algorithm Sanity Check")
        check_my_algorithms()
        
        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        success_count = 0
        total_tests = len(basic_results)
        
        for model_name, result in basic_results.items():
            if result['status'] == 'SUCCESS':
                success_count += 1
                print(f"✅ {model_name}: PASSED")
            else:
                print(f"❌ {model_name}: FAILED - {result['error']}")
        
        print(f"\nOverall: {success_count}/{total_tests} algorithms working")
        
        if success_count == total_tests:
            print("🎉 All tests passed! Ready to run main project!")
        else:
            print("⚠️ Some tests failed - need to debug before main project")
            
    except Exception as e:
        print(f"\n💥 Testing crashed: {e}")
        print("Something is seriously broken - check your imports and data")

if __name__ == "__main__":
    main()