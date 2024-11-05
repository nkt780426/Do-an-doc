import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

with open('roc_2d_siamese.txt', 'r+') as f:
    roc_2d_siamese = f.read()

with open('roc_3d_siamese.txt', 'r+') as f:
    roc_3d_siamese = f.read()

with open('roc_2d_triplet.txt', 'r+') as f:
    roc_2d_triplet = f.read()

with open('roc_3d_triplet.txt', 'r+') as f:
    roc_3d_triplet = f.read()

with open('roc_concat_siamese.txt', 'r+') as f:
    roc_concat_siamese = f.read()

with open('roc_concat_triplet.txt', 'r+') as f:
    roc_concat_triplet = f.read()

with open('roc_2d_classify.txt', 'r+') as f:
    roc_2d_classify = f.read()

with open('roc_3d_classify.txt', 'r+') as f:
    roc_3d_classify = f.read()

with open('roc_concat_classify.txt', 'r+') as f:
    roc_concat_classify = f.read()

def spilt_result_classify(text):
    loss_train = []
    loss_valid = []
    acc_train = []
    acc_valid = []
    roc_train_dis= []
    roc_valid_dis = []    
    roc_train_cos = []
    roc_valid_cos = []
    lines = text.split("\n")
    for line in lines:
        if line.startswith("Train"):
            if len(loss_valid)<len(loss_train):
                loss_train.pop()
                acc_train.pop()
            loss_train.append(float(line.split()[5]))
            acc_train.append(float(line.split()[11]))
        elif line.startswith("Valid"):
            loss_valid.append(float(line.split()[5]))
            acc_valid.append(float(line.split()[11]))
        elif line.startswith("---Epoch:"):
            if 'Train' in line:
                roc_train_dis.append(float(line.split()[6]))
                roc_train_cos.append(float(line.split()[10]))
            elif 'Validation' in line:
                roc_valid_dis.append(float(line.split()[6]))
                roc_valid_cos.append(float(line.split()[10]))
    # print(len(loss_train), len(loss_valid))
    # print(len(acc_train), len(acc_valid))
    # print(len(roc_train_dis), len(roc_valid_dis))
    # print(len(roc_train_cos), len(roc_valid_cos))
    # print()
    return loss_train, loss_valid, acc_train, acc_valid, roc_train_dis, roc_valid_dis, roc_train_cos, roc_valid_cos

def spilt_result(text):
    loss_train = []
    loss_valid = []
    roc_train_dis= []
    roc_valid_dis = []    
    roc_train_cos = []
    roc_valid_cos = []
    lines = text.split("\n")
    for line in lines:
        if line.startswith("Epoch:"):
            if 'Train' in line:
                loss_train.append((float(line.split(' ')[-1])))
            elif 'Validation' in line:
                loss_valid.append((float(line.split(' ')[-1])))
            # if 'Train' in line:
            #     try:
            #         loss_train.append(math.log(float(line.split(' ')[-1])))
            #     except:
            #         loss_train.append(math.log(0.01))
            # elif 'Validation' in line:
            #     try:
            #         loss_valid.append(math.log(float(line.split(' ')[-1])))
            #     except:
            #         loss_valid.append(math.log(0.01))
        elif line.startswith("---Epoch:"):
            if 'Train' in line:
                roc_train_dis.append(float(line.split()[6]))
                roc_train_cos.append(float(line.split()[10]))
            elif 'Validation' in line:
                roc_valid_dis.append(float(line.split()[6]))
                roc_valid_cos.append(float(line.split()[10]))
    # print(len(loss_train), len(loss_valid))
    # print(len(roc_train_dis), len(roc_valid_dis))
    # print(len(roc_train_cos), len(roc_valid_cos))
    # print()
    return loss_train, loss_valid, roc_train_dis, roc_valid_dis, roc_train_cos, roc_valid_cos

# loss_train2, loss_valid2, acc_train, acc_valid2, roc_train_dis2, roc_valid_dis2, roc_train_cos, roc_valid_cos = spilt_result_classify(roc_2d_classify)
# loss_train3, loss_valid3, acc_train, acc_valid3, roc_train_dis3, roc_valid_dis3, roc_train_cos, roc_valid_cos = spilt_result_classify(roc_3d_classify)
loss_train4, loss_valid4, acc_train, acc_valid4, roc_train_dis4, roc_valid_dis4, roc_train_cos, roc_valid_cos = spilt_result_classify(roc_concat_classify)

# loss_train2, loss_valid2, roc_train_dis2, roc_valid_dis2, roc_train_cos, roc_valid_cos = spilt_result(roc_2d_triplet)
# loss_train3, loss_valid3, roc_train_dis3, roc_valid_dis3, roc_train_cos, roc_valid_cos = spilt_result(roc_3d_triplet)
loss_train4, loss_valid4, roc_train_dis40, roc_valid_dis40, roc_train_cos, roc_valid_cos = spilt_result(roc_concat_triplet)

data1 = roc_train_dis4[:100:]
data2 = roc_valid_dis4[:100:]
data3 = roc_train_dis40[:100:]
data4 = roc_valid_dis40[:100:]
data = {
    'Epoch': range(0, len(data1)*10, 10),
    'Train Classification': data1,
    'Valid Classification': data2,
    'Train Triplet': data3,
    'Valid Triplet': data4,
}

df = pd.DataFrame(data)

# Melt DataFrame để có thể vẽ biểu đồ
df_melted = pd.melt(df, id_vars=['Epoch'], var_name='Type', value_name='Value')


sns.set_context("talk")
# sns.set(font_scale=2) 

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Value', hue='Type', data=df_melted)
plt.title('AUC Metric Concat Model')
plt.xlabel('Epoch')
plt.ylabel('Value')
# plt.legend(title='Metrics')
plt.show()




# # data1 = roc_train_dis4[:100]
# # data2 = roc_valid_dis4[:100]
# # data = {
# #     'Epoch': range(9, len(data1)*10, 10),
# #     'Train': data1,
# #     'Valid': data2,
# # }

# # df = pd.DataFrame(data)

# # # Melt DataFrame để có thể vẽ biểu đồ
# # df_melted = pd.melt(df, id_vars=['Epoch'], var_name='Type', value_name='Value')

# # # Vẽ biểu đồ
# # plt.figure(figsize=(10, 6))
# # sns.lineplot(x='Epoch', y='Value', hue='Type', data=df_melted)
# # plt.title('AUC Triplet Concat Model')
# # plt.xlabel('Epoch')
# # plt.ylabel('Value')
# # # plt.legend(title='Metrics')
# # plt.show()


# loss_train, loss_valid, roc_train_dis, roc_valid_dis, roc_train_cos, roc_valid_cos = spilt_result(roc_concat_triplet)

# def find_min_value_and_index(lst):
#     min_value = min(lst)
#     min_index = lst.index(min_value)
#     return min_value, min_index

# def find_max_value_and_index(lst):
#     max_value = max(lst)
#     max_index = lst.index(max_value)
#     return max_value, max_index

# # Tìm min loss cho tập huấn luyện (train) và kiểm tra (valid)
# min_loss_train, min_loss_train_index = find_min_value_and_index(loss_train)
# min_loss_valid, min_loss_valid_index = find_min_value_and_index(loss_valid)

# # Tìm max roc cho tập huấn luyện (train) và kiểm tra (valid)
# max_roc_train, max_roc_train_index = find_max_value_and_index(roc_train_dis)
# max_roc_valid, max_roc_valid_index = find_max_value_and_index(roc_valid_dis)

# # In kết quả
# print(f"Train Loss: Min Loss = {min_loss_train} at index {min_loss_train_index}")
# print(f"Valid Loss: Min Loss = {min_loss_valid} at index {min_loss_valid_index}")
# print(f"Train ROC: Max ROC = {max_roc_train} at index {max_roc_train_index}")
# print(f"Valid ROC: Max ROC = {max_roc_valid} at index {max_roc_valid_index}")