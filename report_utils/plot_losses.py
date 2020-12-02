import pandas
import matplotlib.pyplot as plt
train = "./data/frcnn_1/run-train-tag-Loss_total_loss.csv"
eval = "./data/frcnn_1/run-eval-tag-Loss_total_loss.csv"

STEP = "Step"
LOSS = "Value"


def main():
    # training data
    df = pandas.read_csv(train)
    rolling = 101
    df['Rolling'] = df.Value.rolling(rolling, 10).mean()
    df['epoch'] = df.Value.rolling(rolling).mean()
    df_epoch = df.iloc[::rolling, :]
    print(df)

    x = df[STEP]
    y = df[LOSS]
    plt.plot(x, y, label="Train (raw)")
    plt.plot(x, df['Rolling'], label=f"Train (Rolling avg of {rolling})")
    plt.plot(df_epoch[STEP], df_epoch['epoch'], label=f"Train (averaged)")

    # validation data
    df = pandas.read_csv(eval)
    x = df[STEP]
    y = df[LOSS]
    plt.plot(x, y, label="Val (raw)")

    # general
    plt.xlim(4490)
    plt.legend()
    plt.xlabel("Train steps")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    main()