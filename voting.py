import numpy as np
import os
from sklearn.metrics import roc_auc_score, f1_score


def vote(txt_file_path, vote_type = "soft"):
    
    data = {}

    y_true =[]
    y_pred =[]
    y_prob =[]
    file_name = os.path.basename(txt_file_path)

    with open(txt_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(', ')
            path = parts[0].split(':')[1]
            pred = float(parts[1].split(':')[1])
            label = int(parts[2].split(':')[1])
            
            # path[:-6]를 키로 사용하여 딕셔너리에 데이터 추가
            key = path[:-6]
            if key not in data:
                data[key] = {'pred': [], 'label': []}  # 각 키마다 빈 리스트를 값으로 가짐
            
            # 각 키에 pred와 label 값을 추가
            data[key]['pred'].append(pred)
            data[key]['label'].append(label)   
    for path, info in data.items():
        #print(f"Path: {path}, Pred: {info['pred']}, Label: {info['label']}")
        label = int(sum(info['label'])/len(info['label']))
        if vote_type =="hard":
            #hard voting
            pred = [int(pred + 0.5) for pred in info['pred']]
            threshold = (len(info['label']))/2
            count = sum(pred)
            voting_result = 1 if count >= threshold else 0
        else:
            #soft voting
            avg_pred = sum(info['pred']) / len(info['pred'])
            voting_result = 1 if int(avg_pred + 0.5) else 0
        
        #결과확인
        y_pred.append(voting_result)
        y_true.append(label)
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    num_total = len(data)
    num_correct = np.sum(y_pred == y_true)
    test_acc = (num_correct/ num_total) * 100
    #print(round(test_acc,4))
    auc_score = roc_auc_score(y_true,y_pred )
    f1 = f1_score(y_true,y_pred)
    return test_acc , auc_score, f1
    
#txt_file_path = f'/home/gunwoo/kunwoolee/DEEPFAKE_project/AudioDeepFakeDetection/txt_file/TSSD_wave_6_2ch_prediction.txt'
def main():
    txt_file_dir = f'/home/gunwoo/kunwoolee/DEEPFAKE_project/AudioDeepFakeDetection/txt_file'
    voting_type = ["hard", "soft"]
    result_acc = {}

    for txt_file in os.listdir(txt_file_dir):
        txt_file_path = os.path.join(txt_file_dir, txt_file)
        for vote_type in voting_type:
            test_acc ,auc_score, f1 = vote(txt_file_path, vote_type= vote_type)
            result_acc[f"{txt_file}_{vote_type}"] = {'vote_type' : vote_type, 'test_acc': test_acc, 'f1': f1,'auc': auc_score}
            
    with open(f'vote_result/result_acc', 'w') as file:
        #sorted_paths = sorted(results.keys())
        for key in result_acc:
            value = result_acc[key]
            file.write(f"file_name: {key},vote_tpye: {value['vote_type']} ,test_acc: {value['test_acc']:.4f},f1: {value['f1']:.4f},auc: {value['auc']:.4f}\n")
    file.close()

if __name__ == "__main__":
    main()
 