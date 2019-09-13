import matplotlib.pyplot as plt
import numpy as np

def visualize_result(results, suptitle):
    plt.figure(figsize=(5 * len(results), 5))
    plt.suptitle(suptitle)
    for i, result in enumerate(results):
        plt.subplot(1, len(results), i + 1)
        plt.title(f'Fold {i}')
        plt.plot(result['val_loss'], '-', label=f'validation loss')
        plt.plot(result['loss'], '--', label=f'train loss')
        plt.legend()

def plot_random(anomalies, normal, random_samples=10):
    """
    Plots 2 columns of random anomaly samples and normal samples
    """
    rand_anomalies = np.random.choice(len(anomalies), random_samples, replace=False)
    rand_normal = np.random.choice(len(normal), random_samples, replace=False)
    
    window_size = 5
    
    plt.figure(figsize=(2*window_size, random_samples*window_size))

    for (i, (anom, norm)) in enumerate(zip(anomalies[rand_anomalies], normal[rand_normal])):
        plt.subplot(random_samples, 2, 2*i+1)
        plt.title('Random abnormal sample ')
        plt.plot(anom)

        plt.subplot(random_samples, 2, 2*i+2)
        plt.title('Random normal sample')
        plt.plot(norm)