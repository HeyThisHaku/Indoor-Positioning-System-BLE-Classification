    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import ParameterSampler
    from sklearn.preprocessing import StandardScaler

    # Ruang parameter yang diuji saat itu:
    rf_param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }

    # Diambil 20 kombinasi acak (karena jika full grid search terlalu berat/lama)
    np.random.seed(42)
    param_list = list(ParameterSampler(rf_param_grid, n_iter=20, random_state=42))

    best_rf_acc = 0
    best_rf_params = None

    for i, params in enumerate(param_list):
      clf = RandomForestClassifier(
          class_weight='balanced', random_state=42, n_jobs=-1, **params
      )
      clf.fit(X_train, y_train)
      acc = accuracy_score(y_test, clf.predict(X_test))
      print(f'[{i+1}/20] {params} -> Acc: {acc:.4f}')
      if acc > best_rf_acc:
        best_rf_acc = acc
        best_rf_params = params

    print(f'\n-> Best RF Accuracy: {best_rf_acc:.4f}')
    print(f'-> Best Params: {best_rf_params}')