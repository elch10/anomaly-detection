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

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.legend()

def plot_random(anomalies, normal, random_samples=10):
    """
    Plots 2 columns of random abnormal and normal samples
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


from bokeh.plotting import figure, output_file, show

def ratios_plot(ratios, peaks, anom_start_idxs=None, anomaly_length=None, window_width=None):
    p = figure(plot_width=800, plot_height=600, tools="xwheel_zoom,pan,box_zoom,reset")

    p.line(np.arange(ratios.shape[0]), ratios, line_width=2)
    p.scatter(peaks, ratios[peaks], fill_color="red", size=15)
    
    if anom_start_idxs is not None:
        for start_idx in anom_start_idxs[:, 0]:
            start_idx -= window_width
            r = start_idx + anomaly_length
            if start_idx < 0:
                r += start_idx
                start_idx = 0
            x = np.arange(start_idx, start_idx+anomaly_length)
            p.line(x, ratios[x], line_width=2, color="red")

    show(p)