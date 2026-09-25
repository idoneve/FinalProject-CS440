import sys
import threading
import time

import pandas as pd


def loading_animation(stop):
    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop.is_set():
        sys.stdout.write(
            f"\r{spinners[i]} Attempting to connect to Hugging Face and load SpecMine dataset... (This will take a minute)"
        )
        sys.stdout.flush()
        i = (i + 1) % len(spinners)
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 95 +"\r")
    sys.stdout.flush()


def load_data():
    stop = threading.Event()

    spinner_thread = threading.Thread(target=loading_animation, args=(stop,))
    spinner_thread.daemon = True
    spinner_thread.start()

    try:
        df = pd.read_parquet(
            "hf://datasets/ShyAgarwal/specmine/data/spec_files.parquet",
            engine="fastparquet",
        )
        return df
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Check requirements.txt for needed packages")
        return None
    finally:
        stop.set()
        spinner_thread.join()


if __name__ == "__main__":
    start_time = time.time()
    df = load_data()
    if df is not None:
        print(
            f"\nDataset Loaded successfully in {time.time() - start_time:.2f} seconds"
        )
        print(f"Total Rows: {len(df)}")
        print("\nDataset Columns:")
        print(df.columns.tolist())
        print("\nFirst 5 Rows Preview:")
        print(df.head())
