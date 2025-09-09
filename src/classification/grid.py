from sklearn.model_selection import GridSearchCV

def ADNI_run_gridsearch(train, classifiers, param_grids, cv=5, scoring='balanced_accuracy'):
    """
    Runs GridSearchCV on multiple classifiers with their respective parameter grids.

    Parameters
    ----------
    train : DataFrame 
        Training set
    classifiers : dict
        Dictionary with model names as keys and classifier objects as values.
        Example: {"Random Forest": RandomForestClassifier(), "LogReg": LogisticRegression()}
    param_grids : dict
        Dictionary with model names as keys and parameter grids as values.
        Example: {"Random Forest": {"n_estimators": [100, 200]}, "LogReg": {"C": [0.1, 1, 10]}}
    cv : int, default=5
        Number of folds for cross-validation.
    scoring : str, default='f1_weighted'
        Scoring metric to optimize (e.g., 'accuracy', 'roc_auc', 'f1_macro').

    Returns
    -------
    best_models : dict
        Dictionary containing best estimator, parameters, and score for each classifier.
    """
    # Target column
    y_train = train['DX']
    X_train = train.drop(columns=['DX'])
    
    best_models = {}
    errors = {}
    
    for name, clf in classifiers.items():
        print(f"\nRunning GridSearch for {name} ...")
        param_grid = param_grids.get(name, {})
        
        grid = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            error_score='raise'  
        )
        
        try:
            grid.fit(X_train, y_train)
            best_models[name] = {
                "best_estimator": grid.best_estimator_,
                "best_params": grid.best_params_,
                "best_score": grid.best_score_
            }
            print(f"Best params for {name}: {grid.best_params_}")
            print(f"Best {scoring}: {grid.best_score_:.4f}")
        
        except Exception as e:
            print(f"Classifier {name} failed: {e}")
            errors[name] = str(e)
    
    return best_models

