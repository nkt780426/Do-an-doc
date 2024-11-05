import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, save_best_loss=False, path_save='', num_epoch_roc = -1, train_loader_image=None, val_loader_image=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    best_loss = 1000
    best_model = None

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        if save_best_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict().copy()
        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        if num_epoch_roc != -1:
            if (epoch + 2) % num_epoch_roc == 0:
                roc_auc_dis, roc_auc_cos = check_ROC(train_loader_image, model, cuda)
                print('---Epoch: {}/{}. Train set: Distance ROC: {:.4f}\tCosine similarity ROC: {:.4f}'
                    .format(epoch + 1, n_epochs, roc_auc_dis, roc_auc_cos))
                roc_auc_dis, roc_auc_cos = check_ROC(val_loader_image, model, cuda)
                print('---Epoch: {}/{}. Validation set: Distance ROC: {:.4f}\tCosine similarity ROC: {:.4f}'
                    .format(epoch + 1, n_epochs, roc_auc_dis, roc_auc_cos))
    if save_best_loss:
        torch.save(best_model, path_save)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in (enumerate(train_loader)):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        # if batch_idx % log_interval == 0:
        #     message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         batch_idx * len(data[0]), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), np.mean(losses))
        #     for metric in metrics:
        #         message += '\t{}: {}'.format(metric.name(), metric.value())

        #     print(message)
        #     losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics

def check_ROC(val_loader, model, device):
    with torch.no_grad():
        model.eval()
        outputs_list = []

        for batch in val_loader:
            images, ids = batch  # Giả sử batch trả về (ảnh, nhãn)
            images = images.cuda()
            outputs = model.get_embedding(images)
            
            for i in range(outputs.size(0)):
                outputs_list.append((ids[i].item(), outputs[i].cpu().numpy()))  # Lưu ID và output
        scores_dis = []
        labels_dis = []
        scores_cos = []
        labels_cos = []

        for i in range(len(outputs_list)):
            for j in range(i + 1, len(outputs_list)):
                id1, tensor1 = outputs_list[i]
                id2, tensor2 = outputs_list[j]

                # Chuyển đổi numpy array về tensor và chuyển lên thiết bị cuda
                tensor1 = torch.tensor(tensor1)
                tensor2 = torch.tensor(tensor2)

                # Tính khoảng cách Euclidean
                score_dis = F.pairwise_distance(tensor1, tensor2).item()
                # Tính cosine similarity
                score_cos = F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)).item()

                # Lưu độ tương đồng và nhãn
                scores_dis.append(score_dis)
                labels_dis.append(0 if id1 == id2 else 1)
                scores_cos.append(score_cos)
                labels_cos.append(1 if id1 == id2 else 0)

        y_true_dis = np.array(labels_dis)
        y_scores_dis = np.array(scores_dis)
        fpr_dis, tpr_dis, thresholds_dis = roc_curve(y_true_dis, y_scores_dis)
        roc_auc_dis = auc(fpr_dis, tpr_dis)

        y_true_cos = np.array(labels_cos)
        y_scores_cos = np.array(scores_cos)
        fpr_cos, tpr_cos, thresholds_cos = roc_curve(y_true_cos, y_scores_cos)
        roc_auc_cos = auc(fpr_cos, tpr_cos)
    return roc_auc_dis, roc_auc_cos