import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import BernoulliNB
import os
import sys
import mlflow
import matplotlib
import pandas as pd

matplotlib.use("Agg")

def parse_args():
    parser = argparse.ArgumentParser(description="BernoulliNB")
    parser.add_argument(
        "--model_param1",
        type=float,
        default=1.0,
        help="(default: 1.0)",
    )
    parser.add_argument(
        "--train_path",
        type=str
    )
    return parser.parse_args()

def main():
    args = parse_args()

    numeric_features = ["if" + str(i) for i in range(1,14)]
    fields = ["id", "label"] + numeric_features + ["cf" + str(i) for i in range(1,27)] + ["day_number"]
    categorical_features = ['cf6', 'cf9', 'cf13', 'cf16', 'cf17', 'cf19', 'cf25', 'cf26']

    # prepare train and test data
    df = pd.read_table(args.train_path, sep='\t', names=fields, index_col=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df[numeric_features + categorical_features],
        df.iloc[:, 1],
        test_size=0.2,
        random_state=42
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        params = {
            'alpha': args.model_param1
        }
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('label', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('BNB', BernoulliNB(**params))
        ])
        model.fit(X_train, y_train)


        # evaluate model
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics
        mlflow.log_metrics({"log_loss": loss, "accuracy": acc})

if __name__ == "__main__":
    main()
    
