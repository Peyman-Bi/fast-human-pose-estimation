import numpy as np


def PCK_metric(predicted_joints, true_joints, Threshold = 0.5):
    """
    0.  Right ankle
    1.  Right knee
    2.  Right hip
    3.  Left hip
    4.  Left knee
    5.  Left ankle
    6.  Right wrist
    7.  Right elbow
    8.  Right shoulder
    9.  Left shoulder
    10. Left elbow
    11. Left wrist
    12. Neck
    13. Head top
    """
    # Calculate True Distance of the head bone link
    HeadBone_distance = np.linalg.norm(true_joints[:,12]-true_joints[:,13])
    correct_parts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    #Claculate Distance between True Joints and Predicted Joints in each Limb
    for i in range(14):
        joint_distance = np.linalg.norm(true_joints[:,i]-predicted_joints[:,i])
        if joint_distance <= (Threshold*HeadBone_distance):
            correct_parts[i] = 1

    return correct_parts


def correct_percentage(predicted_joint, true_joint):
    total_correct_percentage = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    for i in range(len(predicted_joint)):
        correct_part = PCK_metric(predicted_joint[i], true_joint[i])
        total_correct_percentage = np.array(total_correct_percentage) + np.array(correct_part)

    return total_correct_percentage



def print_function(acc, epochs, joint_names, validation=False):
    Final_acc = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    Final_acc[0] = (acc[0]+acc[5])/2.0    #Ankle
    Final_acc[1] = (acc[1]+acc[4])/2.0    #Knee
    Final_acc[2] = (acc[2]+acc[3])/2.0    #Hip
    Final_acc[3] = (acc[6]+acc[11])/2.0   #Wrist
    Final_acc[4] = (acc[7]+acc[10])/2.0   #Elbow
    Final_acc[5] = (acc[8]+acc[9])/2.0    #Shoulder
    Final_acc[6] = acc[13]                #Head

    # For Traing Data
    print('\n-------------------------------------------------------------------------------------')
    if validation:
        print('In PCKh@0.5 Evaluation Metric, The output Validation Accuracy for joints In Epoch {}:\n'.format(epochs))
    else:
        print('In PCKh@0.5 Evaluation Metric, The output Training Accuracy for joints In Epoch {}:\n'.format(epochs))

    for i in range(7):
        print('The Accuracy For {}\t{:.2f}'.format(joint_names[i], Final_acc[i]))
    print('\n=======================================================================================\n')

    return Final_acc
