import os
import joblib
import pprint

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.normpath(os.path.join(HERE, '..', 'models', 'Skin Lesion Diagnostic Model', 'skin-diagnosis.joblib'))

def main():
    print('Inspecting:', MODEL_PATH)
    print('Exists:', os.path.exists(MODEL_PATH))
    if not os.path.exists(MODEL_PATH):
        return
    data = joblib.load(MODEL_PATH)
    print('\nTop-level keys:')
    print(list(data.keys()))
    print('\ntraining_history:')
    pprint.pprint(data.get('training_history', '<no training_history>'))
    th = data.get('training_history', {}) or {}
    print('\nExtracted metrics:')
    print('test_accuracy:', th.get('test_accuracy') or th.get('accuracy') or th.get('val_acc') )
    print('validation_accuracy:', th.get('validation_accuracy') or th.get('val_accuracy') or th.get('val_acc'))
    print('roc_auc:', th.get('roc_auc'))
    print('f1_score:', th.get('f1_score'))
    print('precision:', th.get('precision'))
    print('recall:', th.get('recall'))
    print('\ntraining_details:')
    pprint.pprint(data.get('training_details', '<no training_details>'))

if __name__ == '__main__':
    main()
