import pandas as pd
import matplotlib.pyplot as plt

# Load the training history
df = pd.read_csv("../limitred/L0/DeepD3_model.csv")
df3 = pd.read_csv("DeepD3_model_all_DATA.csv")
df2 = pd.read_csv("../limitred/Best/DeepD3_model0.2.5000b.csv")

# Plot the "val_spines_iou_score" column
plt.figure(figsize=(10, 6))
plt.axhline(y = 0.474, color = 'b', linestyle = ':')
plt.ylim(0,1)
plt.plot(df["val_spines_iou_score"])
plt.plot(df2["val_spines_iou_score"])
plt.plot(df3["val_spines_iou_score"])
plt.title("Validation Spines IoU Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.grid(True)
plt.legend(["Reported on paper","Original 5.000", "Triplet loss 5.000", "Original 50.000"])
plt.savefig("Spines.png")
plt.show()
plt.pause(1)


# Plot the "val_spines_iou_score" column
plt.figure(figsize=(10, 6))
plt.axhline(y = 0.6, color = 'b', linestyle = ':')
plt.ylim(0,1)
plt.plot(df["val_dendrites_iou_score"])
plt.plot(df2["val_dendrites_iou_score"])
plt.plot(df3["val_dendrites_iou_score"])
plt.title("Validation Dendrites IoU Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("IoU Score")
plt.grid(True)
plt.legend(["Reported on paper","Original 5.000", "Triplet loss 5.000", "Original 50.000"])
plt.savefig("Spines.png")
plt.show()
plt.pause(1)
