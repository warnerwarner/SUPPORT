# %%
import re

def plotLoss(log_file):

    with open(log_file) as f:
        lines = f.readlines()
    
    pattern = (
        r"Epoch \[(\d+)/(\d+)\] Batch \[(\d+)/(\d+)\] loss : ([\d.]+), loss_l1 : ([\d.]+), loss_l2 : ([\d.]+)"
    )
    epochs = []
    batchs = []
    losses = []
    losses_l1 = []
    losses_l2 = []
    for line in lines:

        match = re.search(pattern, line)
        if match:
            epoch, epoch_total, batch, batch_total, loss, loss_l1, loss_l2 = match.groups()
            epochs.append(int(epoch))
            batchs.append(int(batch))
            losses.append(float(loss))
            losses_l1.append(float(loss_l1))
            losses_l2.append(float(loss_l2))

    fig, ax  = plt.subplots(1, 3, sharey=True)
    losses = np.array(losses)
    epochs = np.array(epochs)
    losses_l1 = np.array(losses_l1)
    losses_l2 = np.array(losses_l2)
    count = 0
    for epoch in np.unique(epochs):
        sub_losses = losses[epochs == epoch]
        sub_l1s = losses_l1[epochs == epoch]
        sub_l2s = losses_l2[epochs == epoch]
        ax[0].plot(np.arange(count, len(sub_losses)+count), sub_losses)
        ax[1].plot(np.arange(count, len(sub_losses)+count), sub_l1s)
        ax[2].plot(np.arange(count, len(sub_losses)+count), sub_l2s)
        count += len(sub_losses)
    ax[1].set_xlabel("Training interval")
    ax[0].set_ylabel('Loss')
    ax[0].set_title("Total loss")
    ax[1].set_title("L1")
    ax[2].set_title("L2")
    return fig, ax, losses, epochs, losses_l1, losses_l2


