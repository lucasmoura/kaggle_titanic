import matplotlib.pyplot as plt


def display_cost_graph(cost_values, legends):
    for cost in cost_values:
        plt.plot(cost)

    plt.legend(legends, loc='upper right')
    plt.show()
