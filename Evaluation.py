from ranx import Qrels, Run, evaluate
import matplotlib.pyplot as plt

# Using ranx software, calculate P@1, P@5, nDCG@5, MRR, and MAP.
# Must MANUALLY ENTER the file name to evaluate
# Author: Abigail Pitcairn
# Version: Dec. 2, 2024

def evaluate_search_result(qrel_file_name, result_file_name):
    # Specify files
    qrel = Qrels.from_file(qrel_file_name, kind="trec")
    run = Run.from_file(result_file_name, kind='trec')

    # Run tests and print results
    print(f'P@1: {evaluate(qrel, run, "precision@1", make_comparable=True)}')
    print(f'P@5: {evaluate(qrel, run, "precision@5", make_comparable=True)}')
    print(f'nDCG@5: {evaluate(qrel, run, "ndcg@5", make_comparable=True)}')
    print(f'MRR: {evaluate(qrel, run, "mrr", make_comparable=True)}')
    print(f'MAP: {evaluate(qrel, run, "map", make_comparable=True)}')

    # ski_jump_data = (evaluate(qrel, run, "precision@5", return_mean=False, make_comparable=True))


# Function to plot the ski jump graph
def plot_ski_jump(data, title="Ski Jump Plot", xlabel="Ranked Queries", ylabel="Precision"):
    # Sort the data in descending order to create the ski jump effect
    sorted_data = sorted(data, reverse=True)

    # Plot the data
    plt.plot(sorted_data, marker='o', linestyle='-', color='b', label='Precision')

    # Add titles and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show grid and legend
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.show()

# Create ski jump plot for p@5
# plot_ski_jump(ski_jump_data)
