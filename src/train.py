pipeline.fit(X_train, y_train)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='average_precision')
    print(f"CV AUPRC: mean={cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Test evaluation
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        'f1': f1_score(y_test, y_pred),
        'auprc': average_precision_score(y_test, y_proba),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    if save_model:
        models_dir = PROJECT_ROOT / 'models'
        models_dir.mkdir(exist_ok=True)
        joblib.dump(pipeline, models_dir / 'xgboost_model.pkl')
        print("XGBoost model saved to models/xgboost_model.pkl")
    
    return pipeline, metrics, X_test, y_test, y_pred, y_proba
