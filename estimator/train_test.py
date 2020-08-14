import numpy as np
from .utils import generate_maps
from .metrics import correct_percentage, print_function
import time
import torch


def train(model, g_model, criterion, optimizer, train_loader, valid_loader,
          num_epochs, device, scheduler, batch_size, joint_detection, joint_names):
    train_loss, valid_loss = [], []
    PCKh_train_accuracy = np.zeros((num_epochs, 7))
    PCKh_valid_accuracy = np.zeros((num_epochs, 7))

    model.train()
    for epoch in range(num_epochs):

        print('Epoch: {}'.format(epoch+1))
        cnt = 0
        tr_total_loss = 0
        val_total_loss = 0
        total_joints = 0
        tr_PCKh_Correct = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        val_PCKh_Correct = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        start_time = time.time()

        for i, (image_name, tr_data, tr_labels) in enumerate(train_loader):
            if cnt % 5 == 0:
                print('.', end='')

            if tr_data.size(0) != batch_size:
              break

            tr_data = tr_data.to(device)
            tr_labels_hm = generate_maps(tr_labels, tr_data.size(-2), tr_data.size(-1), g_model, device).to(device)


            optimizer.zero_grad()
            # tr_output, fianl_pred_heatmaps = model(tr_data)
            tr_output = model(tr_data)
            #print(tr_output.shape)
            fianl_pred_heatmaps = tr_output[:, -15:-1]
            #print(fianl_pred_heatmaps.shape)
            loss_out = criterion(tr_output, tr_labels_hm.repeat(1,4,1,1))

            loss_out.backward()
            optimizer.step()
            scheduler.step()

            tr_total_loss += loss_out
            cnt += 1

            # Calculate PCKh Metric
            #gt_heatmaps = tr_labels_hm.cpu()
            #gt_joints = joint_detection.search(gt_heatmaps)
            #pred_joints = joint_detection.search(fianl_pred_heatmaps.cpu())
            #tr_PCKh_Correct += correct_percentage(pred_joints, gt_joints)
            #total_joints += batch_size
        tr_total_loss /= i
        #-----------------------------------------------------------------------
        #               Start Validation
        #-----------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            for j, (_, val_data, val_labels) in enumerate(valid_loader):
                if val_data.size(0) != batch_size:
                    break

                val_data = val_data.to(device)
                val_labels_hm = generate_maps(val_labels, val_data.size(-2), val_data.size(-1), g_model, device).to(device)

                # val_output, fianl_pred_heatmaps = model(val_data)
                val_output = model(val_data)
                fianl_pred_heatmaps = val_output[:, -15:-1]
                loss_out = criterion(val_output, val_labels_hm.repeat(1,4,1,1))

                val_total_loss += loss_out

                # Calculate PCKh Metric
                gt_heatmaps = val_labels_hm.cpu()
                gt_joints = joint_detection.search(gt_heatmaps)
                pred_joints = joint_detection.search(fianl_pred_heatmaps.cpu())
                val_PCKh_Correct += correct_percentage(pred_joints, gt_joints)
                total_joints += batch_size
        val_total_loss /= j

        print('\t Training Loss: {:.4f}\t Validation Loss: {:.4f}\t Total Time: {:.2f} min'.format(tr_total_loss, val_total_loss, (time.time()-start_time)/60.00))
        train_loss.append(tr_total_loss)
        valid_loss.append(val_total_loss)

        # Show PCKh Percentage for each Joint
        valid_acc = (val_PCKh_Correct/total_joints)*100.00
        final_valid_acc = print_function(valid_acc, epoch+1, joint_names, validation=True)
        PCKh_valid_accuracy[epoch] = final_valid_acc

    return train_loss, valid_loss, PCKh_valid_accuracy
