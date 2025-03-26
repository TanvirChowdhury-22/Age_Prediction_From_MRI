import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from model import *
from sklearn.metrics import r2_score
import io
# Load and preprocess dataset
data_dir    = 'data/test'
label_file  = 'data/gas csv.csv'

image_height    = 128
image_width     = 128
batch_size      = 1


target_size         = (image_height, image_width)
test_dataset        = CustomDataset(data_dir, label_file, target_size,seed_=True, testing=True)
test_loader         = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instantiate the model
# for training change the following line to:
# model = torch.load('model_age_prediction.pt')
model = torch.load('save_model_weights/best_model_convolutional/model_age_prediction.pt')
criterion = nn.MSELoss()
# Validation loop
model.eval()
running_loss = 0.0
loss_values_val = []
predicted_brain_age = np.array([])
chronological_age = np.array([])
with torch.no_grad():
    for images, labels in test_loader:
        images = images.unsqueeze(1).float()  # Add channel dimension
        
        labels = labels.float().unsqueeze(1)
        try: chronological_age = np.append(chronological_age,labels.detach().numpy())
        except: chronological_age = labels.detach().numpy()
        with torch.no_grad():
            outputs = model(images)
        try: predicted_brain_age = np.append(predicted_brain_age,outputs.detach().numpy())
        except: predicted_brain_age = outputs.detach().numpy()
        
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        loss_values_val.append(running_loss)

plt.figure()
plt.plot(np.sort(chronological_age),predicted_brain_age[np.argsort(chronological_age)],'b o')
plt.title("Actual vs Predicted Brain Age")
plt.xlabel("Chronological age (weeks)")
plt.ylabel("Predicted Brain age(weeks)")
plt.savefig('actual_vs_predicted_Loss.png')
plt.grid()
r2 = round(r2_score(chronological_age, predicted_brain_age),3)
print('------------------------------------------------------------------')
print("r2 score = ", r2)
print('------------------------------------------------------------------')
# Perform linear regression to estimate the slope (m) and intercept (b)
m, b = np.polyfit(chronological_age, predicted_brain_age, 1)
# To predict new values using the estimated line
predicted_y = m * np.sort(chronological_age) + b
#######################################################################
plt.plot(np.sort(chronological_age), predicted_y, color='r')
mae =   round(running_loss / len(test_loader.dataset),3)
print(f"Validation Loss: {mae}")
#adding text inside the plot
plt.text(int(np.min(chronological_age))+1, int(np.max(predicted_brain_age)-1), F'MAE = {mae} weeks', fontsize = 12)
plt.text(int(np.min(chronological_age))+1, int(np.max(predicted_brain_age)-2), F'$R^2$ = {r2}', fontsize = 12)
plt.savefig("r2_test_score.png")
#plt.show()

data                            =    pd.read_csv(label_file)
age_data_chronlogical           =    np.sort(chronological_age) #np.sort(chronological_age) #sorted(data['tag_ga'])
age_data_predicted              =    np.copy(predicted_brain_age)


data_distributed                =    dict()


np.random.seed(19680801)
ages = [
    age_data_chronlogical,
    age_data_predicted
]
labels = ['Chronological age (weeks)', 'Predicted Brain age(weeks)']
colors = ['peachpuff', 'orange']

fig, ax = plt.subplots()
ax.set_ylabel('Age (weeks)')

bplot = ax.boxplot(ages,
                   patch_artist=True,  # fill with color
                   labels=labels)  # will be used to label x-ticks

# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.savefig("age_vs_pad_boxplot.png")
plt.show()
