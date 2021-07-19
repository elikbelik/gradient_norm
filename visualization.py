from matplotlib import pyplot as plt


def visualize(history, assume_val=True, figsize=(15,4), plot_diff=False):
    # with validation
    if assume_val:
        f, axs = plt.subplots(1, len(history.history)//2, figsize=figsize)
        idx = 0
        for k in history.history.keys():
            if k.startswith('val_'):
                if not plot_diff:
                    axs[idx].plot(history.epoch, history.history[k[4:]], label='train')
                    axs[idx].plot(history.epoch, history.history[k], label='validation')
                else:
                    axs[idx].plot(history.epoch, 
                                  [x[1]-x[0] for x in zip(history.history[k[4:]], history.history[k])],
                                  label='generalization (val-train)')
                axs[idx].legend()
                axs[idx].set_title(k[4:])
                idx += 1
    
    # no validation
    else:
        f, axs = plt.subplots(1, len(history.history), figsize=figsize)
        for idx, k in enumerate(history.history.keys()):
            axs[idx].plot(history.epoch, history.history[k], label='train')
            axs[idx].set_title(k)
                
    return f,axs

if __name__ == '__main__':
    pass