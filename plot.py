import matplotlib.pyplot as plt


def get_accuracies(file_name):
    accs = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        lines = lines[1:]

        accs = [float(line.strip().split("\t")[-1]) for line in lines]

    return accs

file_names = [
    "logs/ssl_b64_t0.75_mu7.tsv",
    "logs/ssl_b64_t0.85_mu7.tsv",
    "logs/ssl_b64_t0.95_mu7.tsv"
]

names = [
    "t=0.75",
    "t=0.85",
    "t=0.95"
]

for file_name, name in zip(file_names, names):
    accs = get_accuracies(file_name)

    plt.plot(range(len(accs)), accs, label=name)

plt.legend()

plt.savefig("test.png")