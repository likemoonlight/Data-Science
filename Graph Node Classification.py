# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:42:16 2019

@author: jason
"""

import tensorflow as tf
import numpy as np
import csv


##########################################################################################################################################
def Load_train_data():
    train_graph_id_path = "./data/train_graph_id.npy"
    train_links_path = "./data/train_links.npy"
    train_feats_path = "./data/train_feats.npy"
    train_labels_path = "./data/train_labels.npy"
    train_graph_id = np.load(train_graph_id_path)
    train_links = np.load(train_links_path).T
    train_feats = np.load(train_feats_path)
    train_labels = np.load(train_labels_path)
    Graph_20_data = []
    for i in range(20):
        vertex_set = set()
        NodeID_to_index = dict()    
        index_to_NodeID = dict()    
        Graph_20_data.append([vertex_set , NodeID_to_index , index_to_NodeID])
    for i in range(train_graph_id.shape[0]):        
        node_in_what_graph = int(train_graph_id[i]) - 1 
        Graph_20_data[node_in_what_graph][0].add(i)
    for i in range(20):
        A_data = np.zeros(shape=[len(Graph_20_data[i][0]),len(Graph_20_data[i][0])])            
        x_data = np.zeros(shape=[len(Graph_20_data[i][0]) , 50])
        y_data = np.zeros(shape=[len(Graph_20_data[i][0]) , 121])
        Graph_20_data[i].append(A_data)
        Graph_20_data[i].append(x_data)
        Graph_20_data[i].append(y_data)    
        c = 0
        for k in Graph_20_data[i][0]:
            Graph_20_data[i][1][k] = c
            Graph_20_data[i][2][c] = k
            c = c + 1
    for feat_id in range(train_feats.shape[0]):
        feat = train_feats[feat_id]
        label = train_labels[feat_id]  
        feat_in_what_graph = int(train_graph_id[feat_id]) - 1
        feat_in_what_row = int(Graph_20_data[feat_in_what_graph][1][feat_id])    
        Graph_20_data[feat_in_what_graph][4][feat_in_what_row][:] = feat
        Graph_20_data[feat_in_what_graph][5][feat_in_what_row][:] = label 
    for link_id in range(train_links.shape[0]):    
        source = int(train_links[link_id][0])
        target = int(train_links[link_id][1])
        source_in_what_graph = int(train_graph_id[source]) - 1
        target_in_what_graph = int(train_graph_id[target]) - 1    
        assert source_in_what_graph == target_in_what_graph      
        source_in_what_row = int(Graph_20_data[source_in_what_graph][1][source])
        target_in_what_col = int(Graph_20_data[target_in_what_graph][1][target])    
        Graph_20_data[source_in_what_graph][3][source_in_what_row][target_in_what_col] = 1
    for i in range(20):        
        adjacency = Graph_20_data[i][3]
        identity_matrix = np.identity(adjacency.shape[0])
        adjacency_hat = adjacency + identity_matrix       
        summation = np.sum(adjacency_hat , axis = 1)
        half = np.power(summation , 0.5)
        minus_half = 1 / half        
        D_data = np.multiply(summation , identity_matrix)        
        D_data_minus_half = np.multiply(minus_half , identity_matrix)       
        Graph_20_data[i].append(adjacency_hat)
        Graph_20_data[i].append(D_data_minus_half)
        Graph_20_data[i].append(D_data)        
    train_label = []    
    for i in range(20):
        label = Graph_20_data[i][5]
        train_label.append(label)        
    train_label = np.concatenate(train_label , axis=0)        
    positive = np.sum(train_label , axis = 0)
    mean_positive_ratio = positive / train_label.shape[0]    
    label_num = np.sum(train_label , axis = 1)    
    max_label_num = float(np.max(label_num))
    min_label_num = float(np.min(label_num))
    mean_label_num = float(np.sum(label_num / train_label.shape[0]))    
    statistics = {"mean_positive_ratio":mean_positive_ratio,
                  "max_label_num":max_label_num,
                  "min_label_num":min_label_num,
                  "mean_label_num":mean_label_num}
    return Graph_20_data , statistics

def Load_test_data():
    test_graph_id_path = "./data/test_graph_id.npy"
    test_links_path = "./data/test_links.npy"
    test_feats_path = "./data/test_feats.npy"
    test_graph_id = np.load(test_graph_id_path)
    test_links = np.load(test_links_path).T
    test_feats = np.load(test_feats_path)
    Graph_4_data = []
    for i in range(4):
        vertex_set = set()
        NodeID_to_index = dict()    
        index_to_NodeID = dict()    
        Graph_4_data.append([vertex_set , NodeID_to_index , index_to_NodeID])
    for i in range(test_graph_id.shape[0]):
        node_in_what_graph = int(test_graph_id[i]) - 1 - 20
        Graph_4_data[node_in_what_graph][0].add(i)
    for i in range(4):
        A_data = np.zeros(shape=[len(Graph_4_data[i][0]),len(Graph_4_data[i][0])])            
        x_data = np.zeros(shape=[len(Graph_4_data[i][0]) , 50])
        y_data = np.zeros(shape=[len(Graph_4_data[i][0]) , 121])
        Graph_4_data[i].append(A_data)
        Graph_4_data[i].append(x_data)
        Graph_4_data[i].append(y_data)   
        c = 0
        for k in Graph_4_data[i][0]:
            Graph_4_data[i][1][k] = c
            Graph_4_data[i][2][c] = k
            c = c + 1
    for feat_id in range(test_feats.shape[0]):    
        feat = test_feats[feat_id]    
        feat_in_what_graph = int(test_graph_id[feat_id]) - 1 - 20    
        feat_in_what_row = int(Graph_4_data[feat_in_what_graph][1][feat_id])    
        Graph_4_data[feat_in_what_graph][4][feat_in_what_row][:] = feat 
    for link_id in range(test_links.shape[0]):
        source = int(test_links[link_id][0])
        target = int(test_links[link_id][1])
        source_in_what_graph = int(test_graph_id[source]) - 1 - 20
        target_in_what_graph = int(test_graph_id[target]) - 1 - 20    
        assert source_in_what_graph == target_in_what_graph        
        source_in_what_row = int(Graph_4_data[source_in_what_graph][1][source])
        target_in_what_col = int(Graph_4_data[target_in_what_graph][1][target])
        Graph_4_data[source_in_what_graph][3][source_in_what_row][target_in_what_col] = 1
    for i in range(4):       
        adjacency = Graph_4_data[i][3]
        identity_matrix = np.identity(adjacency.shape[0])
        adjacency_hat = adjacency + identity_matrix        
        summation = np.sum(adjacency_hat , axis = 1)
        half = np.power(summation , 0.5)
        minus_half = 1 / half        
        D_data = np.multiply(summation , identity_matrix)        
        D_data_minus_half = np.multiply(minus_half , identity_matrix)
        Graph_4_data[i].append(adjacency_hat)
        Graph_4_data[i].append(D_data_minus_half)
        Graph_4_data[i].append(D_data)    
    return Graph_4_data

def compute(y_pre , y , mean_positive_ratio , label_num):    
    if mean_positive_ratio >= 0.75:
        mean_positive_ratio == 0.75
    elif mean_positive_ratio <= 0.25:
        mean_positive_ratio == 0.25
    y_pre_thres = (y_pre >= float(mean_positive_ratio))    
    P = y.sum()
    TP = (y_pre_thres * y ).sum()
    FP = (y_pre_thres * (1-y)).sum()
    if P == 0 :
        Recall = 0
    else : 
        Recall = TP / P    
    if (TP + FP) == 0:
        Precision = 0
    else:
        Precision = TP / (TP + FP)        
    if (Recall + Precision) == 0:
        return 0
    else:
        return 2 * Recall * Precision / (Recall + Precision) 
    
def W(input_num , output_num , name = None): 
    w = tf.Variable(tf.random.normal([input_num , output_num],mean=0,stddev = 1/np.sqrt(input_num)) , name = name)    
    return w


def B(output_num , name = None):    
    b = tf.Variable(tf.zeros([1, output_num]) + 0.1,)
    return b

def compute_mean(y_pre  , y , statistics):    
    mean_score = 0    
    mean_positive_ratio = statistics["mean_positive_ratio"]
    max_label_num = statistics["max_label_num"]
    min_label_num = statistics["min_label_num"]
    mean_label_num = statistics["mean_label_num"]    
    for i in range(121):
        y_pre_single = (y_pre.T)[i]
        y_single = (y.T)[i]         
        single_class_score = compute(y_pre_single , y_single , mean_positive_ratio[i] , max_label_num)       
        mean_score = mean_score + single_class_score    
    return mean_score / 121

a , _ = Load_train_data()
b = Load_test_data()
train_label = []    
for i in range(20):
    label = a[i][5]
    train_label.append(label)        
train_label = np.concatenate(train_label , axis=0)    
print(train_label.shape)   
positive = np.sum(train_label , axis = 0)
label_num = np.sum(train_label , axis = 1)


def Train():    
    Test_data = Load_test_data()    
    train_Data , statistics = Load_train_data()
    Train_data ,  Valid_data = train_Data[0:] , train_Data[-2:] 
    x = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [1 , None , 50])
    y = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [1 , None , 121])
    A_hat = tf.compat.v1.placeholder(dtype = tf.float32 , shape = [1 , None , None])
    D_minus_half = tf.compat.v1.placeholder(dtype=tf.float32 , shape = [1 , None , None])
    LR = tf.compat.v1.placeholder(tf.float32)
    keep_prob = tf.compat.v1.placeholder(tf.float32)  
    para_size = [50,2048,2048,4096,121]
    W1 = W(para_size[0] , para_size[1] , "Layer_1_W")
    W2 = W(para_size[1] , para_size[2] , "Layer_2_W")
    W3 = W(para_size[2] , para_size[3] , "Layer_3_W")
    W4 = W(para_size[3] , para_size[4] , "Layer_4_W")    
    b1 = B(para_size[1])
    b2 = B(para_size[2])
    b3 = B(para_size[3])
    b4 = B(para_size[4])    
    with tf.compat.v1.variable_scope("GCN_Layer_1"):
        layer_1_1 = tf.matmul(x , W1) + b1  
        layer_1_T = tf.transpose(layer_1_1 , [0,2,1])       
        att_w1 = tf.matmul(layer_1_1 , layer_1_T)
        att_w1 = tf.nn.sigmoid(att_w1)
        att_w1 = tf.multiply(A_hat , att_w1)      
        layer_1_2 = tf.matmul(D_minus_half , layer_1_1)
        layer_1_3 = tf.matmul(att_w1 , layer_1_2)
        layer_1_4 = tf.matmul(D_minus_half , layer_1_3) 
        layer_1_act = tf.nn.tanh(layer_1_4)        
        layer_1_act = tf.nn.dropout(layer_1_act , keep_prob = keep_prob)
    with tf.compat.v1.variable_scope("GCN_Layer_2"):
        layer_2_1 = tf.matmul(layer_1_act , W2) + b2
        layer_2_T = tf.transpose(layer_2_1 , [0,2,1])
        att_w2 = tf.matmul(layer_2_1 , layer_2_T)    
        att_w2 = tf.nn.sigmoid(att_w2)
        att_w2 = tf.multiply(A_hat , att_w2)        
        layer_2_2 = tf.matmul(D_minus_half , layer_2_1)
        layer_2_3 = tf.matmul(att_w2 , layer_2_2)
        layer_2_4 = tf.matmul(D_minus_half , layer_2_3)
        layer_2_act = tf.nn.tanh(layer_2_4)
        layer_2_act = tf.nn.dropout(layer_2_act , keep_prob = keep_prob)
    with tf.compat.v1.variable_scope("GCN_Layer_3"):
        layer_3_1 = tf.matmul(layer_2_act , W3)+ b3        
        layer_3_T = tf.transpose(layer_3_1 , [0,2,1]) 
        att_w3 = tf.matmul(layer_3_1 , layer_3_T)
        att_w3 = tf.nn.sigmoid(att_w3)
        att_w3 = tf.multiply(A_hat , att_w3)       
        layer_3_2 = tf.matmul(D_minus_half , layer_3_1)
        layer_3_3 = tf.matmul(att_w3 , layer_3_2)
        layer_3_4 = tf.matmul(D_minus_half , layer_3_3) 
        layer_3_act = tf.nn.tanh(layer_3_4)        
        layer_3_act = tf.nn.dropout(layer_3_act , keep_prob = keep_prob)       
    with tf.compat.v1.variable_scope("GCN_Layer_4"):
        layer_4_1 = tf.matmul(layer_3_act , W4) + b4
        layer_4_act = tf.nn.sigmoid(layer_4_1)
    cost = tf.reduce_mean(-tf.reduce_sum((y * tf.math.log(layer_4_act + 1e-8) + (1 - y) * tf.math.log(1-layer_4_act + 1e-8)), reduction_indices=[1]))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = LR).minimize(cost)       
    initial = tf.compat.v1.global_variables_initializer()
    Saver = tf.compat.v1.train.Saver(max_to_keep = 15) 
    with tf.compat.v1.Session() as sess:
        print("\nStart Train...\n")
        print("Number of parameters : " , np.sum([np.prod(v.get_shape().as_list()) for v in tf.compat.v1.trainable_variables()]) , "\n")
        Tk = 0
        Epoch = 0        
        Saver.restore(sess,"./model/GCN.ckpt")      
        lr = 1e-3  / 10 / 10
        best_performance = 0.50
        nan_count = 0        
        for i in range(Epoch):                      
            start_time = time.time()
            total_cost = 0
            total_node_num = 0                
            prediction_list = []
            ground_truth_list = []            
            for j in range(20):                
                train_x  , train_y , train_A_hat , train_D_minus_half = Train_data[j][4] , Train_data[j][5] , Train_data[j][6] , Train_data[j][7]                
                node_num = train_x.shape[0]                
                train_x = np.reshape(train_x , [1 , node_num , 50])
                train_y = np.reshape(train_y , [1 , node_num , 121])
                train_A_hat = np.reshape(train_A_hat , [1 , node_num , node_num])
                train_D_minus_half = np.reshape(train_D_minus_half , [1 , node_num , node_num])                
                _ , temp_cost , temp_graph_output = sess.run([optimizer , cost , layer_4_act] , 
                                                             feed_dict={x : train_x , y : train_y , LR : lr , keep_prob: 1.0 , 
                                                                        A_hat : train_A_hat , D_minus_half : train_D_minus_half})    
                temp_graph_output = np.reshape(temp_graph_output , [node_num , 121])
                train_y = np.reshape(train_y , [node_num , 121])                
                prediction_list.append(temp_graph_output)
                ground_truth_list.append(train_y)                
                total_node_num = total_node_num + node_num    
                total_cost = total_cost + temp_cost * node_num
            total_prediction = np.concatenate([pred for pred in prediction_list] , axis = 0)
            total_ground_truth = np.concatenate([gt for gt in ground_truth_list] , axis = 0)            
            performance = compute_mean(total_prediction , total_ground_truth , statistics)            
            total_cost = total_cost / total_node_num
            print("Epoch : " , i , " , Cost : " , round(total_cost,7) , " , Performance : " , round(performance , 5))
            prediction_list = []
            ground_truth_list = []            
            for j in range(2):
                valid_x  , valid_y , valid_A_hat , valid_D_minus_half = Valid_data[j][4] , Valid_data[j][5] , Valid_data[j][6] , Valid_data[j][7]
                node_num = valid_x.shape[0]                
                valid_x = np.reshape(valid_x , [1 , node_num , 50])
                valid_y = np.reshape(valid_y , [1 , node_num , 121])
                valid_A_hat = np.reshape(valid_A_hat , [1 , node_num , node_num])
                valid_D_minus_half = np.reshape(valid_D_minus_half , [1 , node_num , node_num])
                temp_cost , temp_graph_output = sess.run([cost , layer_4_act] , 
                                                         feed_dict={x : valid_x , y : valid_y ,  keep_prob : 1 , 
                                                                        A_hat : valid_A_hat , D_minus_half : valid_D_minus_half})   
                temp_graph_output = np.reshape(temp_graph_output , [node_num , 121])
                valid_y = np.reshape(valid_y , [node_num , 121])                
                prediction_list.append(temp_graph_output)
                ground_truth_list.append(valid_y)               
                total_node_num = total_node_num + node_num    
                total_cost = total_cost + temp_cost * node_num           
            total_prediction = np.concatenate([pred for pred in prediction_list] , axis = 0)
            total_ground_truth = np.concatenate([gt for gt in ground_truth_list] , axis = 0)           
            performance = compute_mean(total_prediction , total_ground_truth , statistics)           
            total_cost = total_cost / total_node_num
            print("         , Cost : " , round(total_cost,7) , " , Performance : " , round(performance , 5))                        
            process_time = time.time() - start_time 
            Tk = Tk + (process_time - Tk) / (i + 1 )
            if i % 10 == 0: 
                print("\nAverage Train time  per epoch : " , Tk , "\n")
            if np.isnan(total_cost) == True:
                print("Nan  Warining , don't  save model.\n")
                nan_count += 1
                pass            
            else:                
                if performance >= best_performance:                
                    best_performance = performance
                    Saver.save(sess,"./model_att_sigmoid_V7/10th_GCN.ckpt")            
            if nan_count == 5:
                print("Model has collapsed , stop Train , the stopping epoch is " , i - 5 , "\n")                
                break            
            if i!= 0 and  (i % 2400 ==0 ) :
                Saver.save(sess,"./model_att_sigmoid_V7/9th_%i_GCN.ckpt"%(i))
                print("----------------------------------------------------------------------------------------------")        
                print("\nWriting testing data...\n")        
                st = time.time()        
                prediction_list = []
                for j in range(4):          
                    test_x  , _ , test_A_hat , test_D_minus_half = Test_data[j][4] , Test_data[j][5] , Test_data[j][6] , Test_data[j][7]               
                    node_num = test_x.shape[0]              
                    test_x = np.reshape(test_x , [1 , node_num , 50])
                    test_A_hat = np.reshape(test_A_hat , [1 , node_num , node_num])
                    test_D_minus_half = np.reshape(test_D_minus_half , [1 , node_num , node_num])
                    temp_graph_output = sess.run([layer_4_act] , 
                                                 feed_dict={x : test_x ,  keep_prob : 1 , 
                                                            A_hat : test_A_hat , D_minus_half : test_D_minus_half})  
                    temp_graph_output = np.reshape(temp_graph_output , [node_num , 121])           
                    prediction_list.append(temp_graph_output)
                total_prediction = np.concatenate([pred for pred in prediction_list] , axis = 0)
                mean_pos = np.reshape(statistics["mean_positive_ratio"],[121,1])   
                mean_pos = np.clip(mean_pos , 0.25 , 0.75)
                predict = (total_prediction.T >= mean_pos)       
                predict = predict.T        
                out = open("V7_answer_9th_%i.csv"%(i) , "a" , newline = "")
                csv_writer = csv.writer(out)
                csv_writer.writerow(["Id" , "Predicted"])
                for i in range(predict.shape[0]):
                    if i % 1000 ==0:
                        print(i)        
                    c = np.where(predict[i]==1)            
                    if c[0].shape[0] == 0:
                        row = [str(i) , " "]
                        csv_writer.writerow(row)    
                    else:
                        haha = ""
                        for j in range(c[0].shape[0]):
                            haha = haha + str(c[0][j]) + " "
                        row = [str(i) , haha]
                        csv_writer.writerow(row)
                out.close()
        prediction_list = []
        for j in range(4):           
            test_x  , _ , test_A_hat , test_D_minus_half = Test_data[j][4] , Test_data[j][5] , Test_data[j][6] , Test_data[j][7]              
            node_num = test_x.shape[0]                
            test_x = np.reshape(test_x , [1 , node_num , 50])
            test_A_hat = np.reshape(test_A_hat , [1 , node_num , node_num])
            test_D_minus_half = np.reshape(test_D_minus_half , [1 , node_num , node_num])
            temp_graph_output = sess.run([layer_4_act] , 
                                         feed_dict={x : test_x ,  keep_prob : 1 , 
                                                    A_hat : test_A_hat , D_minus_half : test_D_minus_half})    
            temp_graph_output = np.reshape(temp_graph_output , [node_num , 121])           
            prediction_list.append(temp_graph_output)
        total_prediction = np.concatenate([pred for pred in prediction_list] , axis = 0)
        mean_pos = np.reshape(statistics["mean_positive_ratio"],[121,1])   
        mean_pos = np.clip(mean_pos , 0.25 , 0.75)
        predict = (total_prediction.T >= mean_pos)        
        predict = predict.T
        out = open("080304.csv" , "a" , newline = "")
        csv_writer = csv.writer(out)
        csv_writer.writerow(["Id" , "Predicted"])
        for i in range(predict.shape[0]):
            if i % 1000 ==0:
                print(i)        
            c = np.where(predict[i]==1)            
            if c[0].shape[0] == 0:
                row = [str(i) , " "]
                csv_writer.writerow(row)    
            else:
                haha = ""
                for j in range(c[0].shape[0]):
                    haha = haha + str(c[0][j]) + " "
                row = [str(i) , haha]
                csv_writer.writerow(row)
        out.close()       


if __name__ == "__main__":
    Train()        
