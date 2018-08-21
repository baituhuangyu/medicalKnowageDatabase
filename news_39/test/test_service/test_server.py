# encoding: utf-8
import requests
import json

url = "http://127.0.0.1:8080/department"


def test_department(content):

    rst = requests.post(url, data=json.dumps({"content": content}))
    if rst.status_code == 200:
        print(json.dumps(rst.json(), indent=4, ensure_ascii=False))
    else:
        raise Exception(rst.status_code)


if __name__ == '__main__':
    content = [
        "隐匿性肾炎，你好医生，尿蛋白正常，隐血+号，红细胞20几个，这种病补充提问：是没法医的吗补充提问：有胡桃夹",
        "昨天下午开始感觉喉咙有点干，有点痛，今天还是一样，傍晚开始，全身抹起来都痛，除了四肢。早几天早上被尿憋醒，尿了很久，尿完感觉阴道有点微痛，之后几天尿完都会有点微痛的症状，尿有点黄，是尿毒症吗，还是感冒了",
    ]
    test_department(content)
