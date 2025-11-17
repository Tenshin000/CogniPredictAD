from sklearn.model_selection import GridSearchCV, cross_val_score

def ADNI_run_gridsearch(X_train, y_train, pipelines, param_grids, cv=5, scoring='balanced_accuracy'):
    """
    Runs GridSearchCV on multiple classifiers with their respective parameter grids.
    Ignores classifiers that fail during fitting and continues with the others.

    Parameters
    ----------
    X_train : DataFrame or array
        Training features.
    y_train : Series or array
        Training labels.
    pipelines : dict
        Dictionary with model names as keys and pipeline objects as values.
    param_grids : dict
        Dictionary with model names as keys and parameter grids as values.
    cv : int, default=5
        Number of folds for cross-validation.
    scoring : str, default='balanced_accuracy'
        Scoring metric to optimize.

    Returns
    -------
    best_models : dict
        Dictionary containing best estimator, parameters, and score for each classifier.
    """
    best_models = {}
    errors = {}
    cv_scores = {}
    
    # GridSearch by model
    for name, clf in pipelines.items():
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
            
            # Save fold-by-fold cross-validation scores on the best model
            best_clf = grid.best_estimator_
            scores = cross_val_score(best_clf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            cv_scores[name] = scores
            print(f"{name} CV scores: {scores}")
        
        except Exception as e:
            print(f"Classifier {name} failed: {e}")
            errors[name] = str(e)
    
    return best_models
