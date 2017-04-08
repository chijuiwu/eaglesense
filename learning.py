import argparse
import os
import pickle
import itertools
import random

import eaglesense as es

import xgboost as xgb
from sklearn import model_selection
from sklearn import metrics

import numpy as np
import pandas as pd


class Model(object):
    def __init__(self, output_dname):
        self.output_dname = output_dname
        self.X_csv = es.config.TOPVIEWKINECT_FEATURES_CSV_FNAME.format(id=output_dname)
        self.y_csv = es.config.TOPVIEWKINECT_LABELS_CSV_FNAME.format(id=output_dname)

    def preprocess(self):
        print("Preprocessing data...")

        subjects_list = list()
        for subject_id in next(os.walk(es.config.TOPVIEWKINECT_DATA_DNAME))[1]:
            if not subject_id.isdigit():
                continue
            else:
                subjects_list.append(int(subject_id))
        subjects_list = np.sort(subjects_list)

        X_fo = open(self.X_csv, "w+")
        y_fo = open(self.y_csv, "w+")
        header = True

        for subject_id in subjects_list:
            print("- Subject", subject_id)

            subject_X_csv = es.config.TOPVIEWKINECT_FEATURES_CSV_FNAME.format(id=subject_id)
            subject_X_df = pd.read_csv(subject_X_csv)
            subject_y_csv = es.config.TOPVIEWKINECT_LABELS_CSV_FNAME.format(id=subject_id)
            subject_y_df = pd.read_csv(subject_y_csv)

            if -1 in subject_y_df["activity"].values:
                print("Missing labels in", subject_id, "Exiting...")
                break

            subject_y_df = subject_y_df.loc[subject_y_df["skeleton_id"] == 0]
            subject_y_df = subject_y_df.loc[subject_y_df["activity"] != 6]
            frame_indices = subject_y_df["frame_id"].values

            subject_X_df = subject_X_df.loc[subject_X_df["frame_id"].isin(frame_indices)]
            subject_y_df = subject_y_df.loc[subject_y_df["frame_id"].isin(subject_X_df["frame_id"].values)]

            ignored_X_cols = ["frame_id", "skeleton_id", "x", "y", "z"]
            ignored_y_cols = ["frame_id", "skeleton_id"]

            subject_X_df = subject_X_df.drop(labels=ignored_X_cols, axis=1)
            subject_X_df["subject"] = int(subject_id)
            subject_X_df = subject_X_df.astype(np.float32)
            subject_X_df.to_csv(X_fo, header=header, index=False)

            subject_y_df = subject_y_df.drop(labels=ignored_y_cols, axis=1)
            subject_y_df["subject"] = int(subject_id)
            subject_y_df = subject_y_df.astype(np.int)
            subject_y_df.to_csv(y_fo, header=header, index=False)

            header = False

        X_fo.close()
        y_fo.close()

        print("Done!")

    def initial_cross_subject(self):
        print("Initial cross-subject test...")
        X_df = pd.read_csv(self.X_csv)
        y_df = pd.read_csv(self.y_csv)

        subjects_list = np.unique(y_df["subject"])
        activities_list = np.unique(y_df["activity"])

        train_subject_indices = [subject for subject in subjects_list if subject % 2 == 1]
        test_subject_indices = [subject for subject in subjects_list if subject % 2 == 0]

        X_train_df = X_df[X_df["subject"].isin(train_subject_indices)].reset_index()
        y_train_df = y_df[y_df["subject"].isin(train_subject_indices)].reset_index()
        X_train = X_train_df.drop(labels="subject", axis=1).values
        y_train = y_train_df["activity"].values.ravel()
        train_dmatrix = xgb.DMatrix(X_train, y_train)

        X_test_df = X_df[X_df["subject"].isin(test_subject_indices)].reset_index()
        y_test_df = y_df[y_df["subject"].isin(test_subject_indices)].reset_index()
        X_test = X_test_df.drop(labels="subject", axis=1).values
        y_test = y_test_df["activity"].values.ravel()
        teszt_dmatrix = xgb.DMatrix(X_test, y_test)

        X_all = X_df.drop(labels="subject", axis=1).values
        y_all = y_df["activity"].values.ravel()
        all_dmatrix = xgb.DMatrix(X_all, y_all)

        # Hyperparameter (see /models/)

        booster_params = {
            "learning_rate": 0.05,
            "n_estimators": 386,
            "max_depth": 5,
            "min_child_weight": 1,
            "max_delta_step": 1,
            "gamma": 0.5,
            "subsample": 0.5,
            "colsample_bytree": 0.5,
            "colsample_bylevel": 0.5,
            "reg_lambda": 1,
            "reg_alpha": 0,
            "scale_pos_weight": 1,
            "objective": "multi:softmax",
            "num_class": 6,
            "silent": 0,
            "seed": 42
        }

        booster = xgb.train(params=booster_params, dtrain=train_dmatrix, num_boost_round=booster_params["n_estimators"])
        y_predicted = booster.predict(teszt_dmatrix)

        accuracy = metrics.accuracy_score(y_test, y_predicted)
        cm = metrics.confusion_matrix(y_test, y_predicted)
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm *= 100
        print(accuracy)
        print(cm)

        booster_final = xgb.train(params=booster_params, dtrain=all_dmatrix, num_boost_round=booster_params["n_estimators"])
        booster_final.save_model(es.config.EAGLESENSE_MODEL_FNAME.format(model="init-cs"))

        print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Run the EagleSense learning module"
    )

    parser.add_argument(
        "--preprocess", dest="preprocess", action="store_true",
        help="Preprocessing"
    )

    parser.add_argument(
        "--train", dest="train", type=str,
        help="Training"
    )

    parser.add_argument(
        "--output", dest="output", type=str,
        help="Output directory"
    )

    args = parser.parse_args()

    model = Model(output_dname=args.output)

    if args.preprocess:
        model.preprocess()

    if args.train == "all":
        model.initial_cross_subject()


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    main()
