import json

def read_json(file_path):
    """
    读取指定路径的 JSON 文件并返回数据。
    
    :param file_path: 要读取的 JSON 文件的路径
    :return: 从 JSON 文件中读取的数据
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def write_to_json(data, file_path):
    """
    将数据写入指定路径的 JSON 文件。
    
    :param data: 要写入文件的数据
    :param file_path: 要写入的 JSON 文件的路径
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__" :
    read_path = "./old/medqa_d2n_train.json"
    data = read_json(read_path)
    
    write_json = []
    # print(type(data))
    # print(data[0]["input"].split('*****KNOWLEDGE*****')[1])
    for js in data:
        write_js = {}
        write_js['input'] = js['input'].split('*****KNOWLEDGE*****')[0]
        write_js['output'] = js['output']
        write_js['rationale'] = js['rationale']
        write_json.append(write_js)
    # print(write_json)
        
    write_path = './medqa_d2n_train.json'
    write_to_json(write_json, write_path)