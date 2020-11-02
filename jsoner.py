import io
from config import *

path = '/Users/tal/Dropbox/Projects/Facial_Attributes_Detection/json'
sum_dict = {'name': [], 'result': [], 'best': {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}}
for file in sorted(os.listdir(path)):
    sum_dict['name'].append(file.split('.')[0])
    json_path = os.path.join(path, file)
    history = json.load(io.open(json_path))
    sum_dict['result'].append(history)
    for k, v in history.items():
        if k.endswith('loss'):
            sum_dict['best'][k].append(min(v))
        else:
            sum_dict['best'][k].append(max(v))

df = pd.DataFrame({'name': sum_dict['name'], 'loss': sum_dict['best']['loss'], 'val_loss': sum_dict['best']['val_loss'],
                   'accuracy': sum_dict['best']['accuracy'], 'val_accuracy': sum_dict['best']['val_accuracy']})

name = df['name'].apply(lambda x: x.split('_')[0]).unique()
means_acc, means_loss, means_vloss, means_vacc = [], [], [], []
means = [means_acc, means_loss, means_vloss, means_vacc]

for n in name:
    # ax = plt.subplot(1, 4, i)
    means_acc.append(df['accuracy'][df['name'].str.startswith(n)].mean())
    means_vacc.append(df['val_accuracy'][df['name'].str.startswith(n)].mean())
    means_loss.append(df['loss'][df['name'].str.startswith(n)].mean())
    means_vloss.append(df['val_loss'][df['name'].str.startswith(n)].mean())
i = 1
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, i)
sns.barplot(name, means_vacc, ci="sd")
plt.title('Validation Acc.')
i += 1
plt.subplot(1, 2, i)
sns.barplot(name, means_vloss, data=df, ci="sd")
plt.title('Validation Loss.')
plt.show()

# err = []
# for n in name:
#     means.append(df[df['name'].str.startswith(n), 1:].mean())
#     err.append(df[df['name'].str.startswith(n), 1:].std())


# for name in sum_dict['name']:
#     plt.bar(name, sum_dict['best']['loss'])
#     plt.show()

# print(sum_dict['best'])
# for name, value in sum_dict.items():
#     min_loss = sum_dict[name]['result']
#     print(min_loss)

# Plot best model
# sns.barplot(x=sum_dict['name'], y=sum_dict['best'].values(), color='m')
# plt.title('Machine Learning Algorithm Accuracy Score \n')
# plt.xlabel('Accuracy Score (%)')
# plt.ylabel('Algorithm')
# name_best_model = df['MLA Name'].values[0]
# plt.show()
