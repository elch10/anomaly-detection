from sklearn.model_selection import TimeSeriesSplit

def cross_validate(model_fn, n_splits, fit_params, X, y):
    tssplit = TimeSeriesSplit(n_splits=3)
    results = []

    for i, (train_idx, test_idx) in enumerate(tssplit.split(X, y)):
        print(f'Fold {i}...')
        model = model_fn()
        history = model.fit(X[train_idx],
                            y[train_idx],
                            validation_data=(X[test_idx], y[test_idx]),
                            **fit_params)
        results.append(history.history)
    
    return results