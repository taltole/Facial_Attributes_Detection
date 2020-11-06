import io
from config import *


def find_best_model(path, label='All'):
    """
    :param path: path to the forder that contains all the json files
    :param label: label to consider to find the best model. The default value 'All' will
                return the best model in general for all attributes
    """
    sum_dict = {'name': [], 'result': [], 'best': {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}}
    label = label.lower()
    if label == 'all':
        list_file = [file for file in os.listdir(path)]
    else:
        list_file = [file for file in os.listdir(path) if label in file.lower()]
    for file in sorted(list_file):
        sum_dict['name'].append(file.split('.')[0])
        json_path = os.path.join(path, file)
        history = json.load(io.open(json_path))
        sum_dict['result'].append(history)

        # finding best params
        for k, v in history.items():
            if k.endswith('loss'):
                sum_dict['best'][k].append(min(v))
            else:
                sum_dict['best'][k].append(max(v))

    df = pd.DataFrame(
        {'name': sum_dict['name'], 'loss': sum_dict['best']['loss'], 'val_loss': sum_dict['best']['val_loss'],
         'accuracy': sum_dict['best']['accuracy'], 'val_accuracy': sum_dict['best']['val_accuracy']})

    name = df['name'].apply(lambda x: x.split('_')[0]).unique()
    # if x.count('_') <= 2 else '_'.join(x.split('_')[:2])).unique()
    means_acc, means_loss, means_vloss, means_vacc = [], [], [], []
    # means = [means_acc, means_loss, means_vloss, means_vacc]
    df_mean = pd.DataFrame()
    for n in name:
        df_mean['name'] = n
        means_acc = df['accuracy'][df['name'].str.startswith(n)].mean()
        means_vacc = df['val_accuracy'][df['name'].str.startswith(n)].mean()
        means_loss = df['loss'][df['name'].str.startswith(n)].mean()
        means_vloss = df['val_loss'][df['name'].str.startswith(n)].mean()
        df_mean.loc[df_mean['name'] == n, 'means_vacc'] = means_vacc
        df_mean.loc[df_mean['name'] == n, 'means_acc'] = means_acc
        df_mean.loc[df_mean['name'] == n, 'means_vloss'] = means_vloss
        df_mean.loc[df_mean['name'] == n, 'means_loss'] = means_loss

    i = 1
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, i)
    sns.barplot(df_mean['means_vacc'], df_mean['name'], data=df_mean.sort_values(by='means_vacc'), ci="sd", orient='h')
    plt.title('Validation Acc.')
    i += 1
    plt.subplot(1, 2, i)
    sns.barplot(df_mean['means_vloss'], df_mean['name'], data=df_mean.sort_values(by='means_vloss'), ci="sd",
                orient='h')
    plt.title('Validation Loss.')
    plt.show()


def summarize_classic_cls(csv_path, att, model):
    acc_list = [i for i in os.listdir(csv_path) if i.startswith('sum')]
    df_dict = pd.DataFrame()
    for file in acc_list:
        temp_df = pd.read_csv(os.path.join(csv_path, file))
        name = file.split('_')[2:-1]
        name = ['_'.join(name) if len(name) >= 2 else ''.join(name)]
        temp_df.iloc[:, 0] = name
        df_dict = df_dict.append(temp_df)
    df_dict.rename(columns={'Unnamed: 0': 'Attribute', 'MLA Name': 'Classifier', 'MLA Test Accuracy Mean': 'Acc',
                            'Run Time': 'Run_Time', 'MLA pred': 'y_pred'}, inplace=True)
    df = df_dict.sort_values(by='Acc')
    print(df)

    #  Classifiers Comparison
    if model is not None:
        yaxis = df['Attribute'][df['Classifier'] == model]
        xaxis = df['Acc'][df['Classifier'] == model].apply(lambda x: x.mean() if not isinstance(x, float) else x)
        title = model.title() + ' Comparison'
        # if att is not None:
        #     return f"{model} Acc mean score for {att}:\t{np.where(((df['Classifier'] == model) & (df['Attribute'] == att)), [df['Acc'].mean() if not isinstance(df['Acc'].values, float) else df['Acc'].values])}"
    else:
        yaxis = df['Attribute']
        xaxis = df['Acc']
        title = 'Mean Acc Score Comparison'

    #  Attribute Comparison
    if att is not None:
        yaxis = df['Classifier'][df['Attribute'] == att]
        xaxis = df['Acc'][df['Attribute'] == att].apply(lambda x: x.mean() if not isinstance(x, float) else x)
        title = att.title() + ' Comparison'
        # if model is not None:
        # return f"{att} Acc mean score for {model}:  " \
        #        f"{np.where(((df['Classifier'] == model) & (df['Attribute'] == att)), df['Acc'].mean())}"
    else:
        yaxis = df['Attribute']
        xaxis = df['Acc']
        title = 'Mean Acc Score Comparison'
    hue = df['Classifier'].unique()

    # Plot
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 1, 1)
    sns.barplot(xaxis,
                yaxis,
                # hue=hue,
                # hue=df['Classifier'].unique()v,
                orient='h')
    plt.title(title)
    plt.xticks(np.arange(0, 1, 0.05))
    plt.show()


if __name__ == '__main__':
    label = 'Hat'
    find_best_model(PATH_JSON)
    summarize_classic_cls(PATH_CSV, att=None, model=None)
