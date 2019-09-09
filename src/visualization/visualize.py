import matplotlib.pyplot as plt

def visualize_result(results, suptitle):
    plt.figure(figsize=(5 * len(results), 5))
    plt.suptitle(suptitle)
    for i, result in enumerate(results):
        plt.subplot(1, len(results), i + 1)
        plt.title(f'Fold {i}')
        plt.plot(result['val_loss'], '-', label=f'validation loss')
        plt.plot(result['loss'], '--', label=f'train loss')
        plt.legend()