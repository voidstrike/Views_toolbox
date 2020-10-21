import json

# Dummy problemDoc.json
def createDummyProblemDoc(data_name, target_tuple, task_type, metric):
    res_dict = dict()
    about_block = {'problemID': data_name,
            'problemName': 'NULL',
            'problemVersion': '4.0.0',
            'problemSchemaVersion': '4.0.0',
            'taskKeywords': [item for item in task_type]}
    
    # DATA_BLOCK is a list of dict
    inner_dict = {'datasetID': data_name,
            'targets':[
                {'targetIndex': idx, 'resID': 'learningData', 'colIndex': tgt_idx, 'colName': tgt_name}
                for idx, (tgt_name, tgt_idx) in enumerate(target_tuple)]}
    data_block = [inner_dict]

    # DATA_SPLIT_BLOCK -- USELESS
    data_split_block = {'method': 'holdout', 'testSize': 0.2, 'stratified': True, 'numRepeats': 0, 'randomSeed': 42, 'splitsFile': 'dataSplits.csv',
            'datasetViewMaps':{
                'train': [{'from': data_name, 'to': f'{data_name}_TRAIN'}],
                'test': [{'from': data_name, 'to': f'{data_name}_TEST'}],
                'score': [{'from': data_name, 'to': f'{data_name}_SCORE'}]}}
    
    # PERFORMANCE BLOCK
    performance_block = [{'metric': metric}]
    input_block = {'data':data_block, 'dataSplits': data_split_block, 'performanceMetrics': performance_block}

    # EXPECTED BLOCK
    output_block = {'predictionsFile': 'predictions.csv'}

    res_dict['about'] = about_block
    res_dict['inputs'] = input_block
    res_dict['expectedOutputs'] = output_block
    return res_dict

def createDatasetDoc(data_name, split, x_features, y_features):
    res_dict = dict()
    about_block = {'datasetID': f'{data_name}_{split}',
            'datasetName': 'NULL',
            'license': 'No license',
            'approximateSize': '',
            'datasetSchemaVersion': '4.0.0',
            'redacted': True,
            'datasetVersion': '4.0.0',
            'digest': 'dummyabcdefghijklmnopqrstuvwxyz'}

    data_block = list()
    tmp = {'resID': 'learningData', 'resPath': 'tables/learningData.csv', 'resType': 'table', 'resFormat': {'text/csv': ['csv']}}
    tmp['isCollections'] = False
    tmp['columns'] = list()
    tmp['columns'].append({'colIndex': 0, 'colName': 'd3mIndex', 'colType': 'integer', 'role': ['index']})
    for idx, feature in enumerate(x_features):
        if feature in ['month', 'country_id', 'pg_id', 'country_name']:
            tmp['columns'].append({'colIndex': idx+1, 'colName': feature, 'colType': 'categorical', 'role': ['attribute']})
        else:
            tmp['columns'].append({'colIndex': idx+1, 'colName': feature, 'colType': 'real', 'role': ['attribute']})
    
    assert len(y_features) == 1
    tmp['columns'].append({'colIndex': len(tmp['columns']), 'colName': y_features[0], 'colType': 'real', 'role': ['suggestedTarget']})

    data_block.append(tmp)
    res_dict['about'] = about_block
    res_dict['dataResources'] = [tmp]
    return res_dict
    
def createDataSplit(input_df, mode='TRAIN'):
    ans = [[item, mode, 0, 0] for item in input_df.index]
    head = ['d3mIndex', 'type', 'repeat', 'fold']
    ans.insert(0, head)
    return ans

