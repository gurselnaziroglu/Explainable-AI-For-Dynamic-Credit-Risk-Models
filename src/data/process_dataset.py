"""Main script for preprocessing raw credit risk data."""

import argparse
import glob
import os

import pandas as pd

from src.data.categorical_encoding import encode_categotical_values
from src.data.missing_imputation import handle_missing_values


def parser():
    """Parser function to run arguments from the command line and to add description to sphinx."""
    parser = argparse.ArgumentParser(
        description="""
    Python wrapper to parse the source path of the raw dataset, and destination path of the processed dataset.
    """
    )

    # Add an argument
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="source path of the dataset to be processed",
    )
    parser.add_argument(
        "--dest",
        type=str,
        required=True,
        help="destination path of the processed dataset",
    )

    parser.add_argument(
        "--start",
        type=int,
        required=False,
        help="start year of the data you want to get",
    )

    parser.add_argument(
        "--end",
        type=int,
        required=False,
        help="end year of the data you want to get",
    )
    return parser


def process_dataset_call(src, dest, start, end, dataset_file_name=None):
    saved_file_paths = []
    for dataset_path in glob.glob(os.path.join(src, "*.csv")):
        filename = os.path.basename(dataset_path)

        if dataset_file_name is not None and filename != dataset_file_name:
            continue

        print(f"Processing {filename}:")
        df = pd.read_csv(
            dataset_path, delimiter=";", index_col="time_series", decimal=","
        )

        df = df.drop(columns=["Unnamed: 0"])
        df.index = pd.to_datetime(
            df.index.map(lambda s: f"{str(s)[:-2]}-{str(s)[-2:]}")
        )

        if start is not None:
            df = df.loc[df.index.year >= start]

        if end is not None:
            df = df.loc[df.index.year <= end]

        external_data_source = os.path.join(os.path.dirname(src), "external")

        print("\tHandling Missing Values...")
        df = handle_missing_values(df, external_source=external_data_source)

        print("\tEncoding Categorical Values...")
        df = encode_categotical_values(df, external_source=external_data_source)

        print("\tSaving...")
        filename_no_extension = os.path.splitext(filename)[0]

        min_year = df.index.min().year
        max_year = df.index.max().year

        new_file_path = (
            f"{dest}/{filename_no_extension}_processed[{min_year},{max_year}].csv"
        )
        df.to_csv(new_file_path)
        saved_file_paths.append(new_file_path)

        print(f"New File Saved to {new_file_path}!")
    return saved_file_paths


def main():
    """Entry point for the preprocessing script."""
    args = parser().parse_args()
    process_dataset_call(src=args.src, dest=args.dest, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
