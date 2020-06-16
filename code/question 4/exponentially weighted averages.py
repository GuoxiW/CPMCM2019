import random
import matplotlib.pyplot as plt


def mean_average(data, num):
    """
    exponentially weighted averages with bias correction
    :param data:
    data: a list of data with noise
    num: average how many data in the calculation
    :return: new_data: a list of mean data
    """
    new_data = []
    v = 0
    length = len(data)
    beta = 1 - (1 / num)
    for i in range(length):
        tmp = beta * v + (1 - beta) * data[i]
        v = tmp
        tmp = tmp / (1 - beta ** (i + 1))
        new_data.append(tmp)

    return new_data

if __name__=='__main__':
    data = []
    x = []
    for i in range(100):
        data.append(8 + random.random())
        x.append(i)
    new_data = mean_average(data, 10)
    plt.plot(x, data, 'ro-', label='orl')
    plt.plot(x, new_data, 'bo-', label='mean')
    plt.legend()
    plt.show()
    # print(data)
    # print(mean_average(data, 10))

