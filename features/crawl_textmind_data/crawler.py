# coding:utf-8

"""
爬取文心系统的数据，提取特征
"""

import http.cookiejar
import json
import urllib.request
import urllib.parse
import numpy as np

from features.crawl_textmind_data import input_textmind_data

headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:35.0) Gecko/20100101 Firefox/35.0'}


class Crawler:
    def __init__(self):
        self.cj = http.cookiejar.LWPCookieJar()
        # 将一个保存cookie对象，和一个HTTP的cookie的处理器绑定
        self.cookie_processor = urllib.request.HTTPCookieProcessor(self.cj)
        # 创建一个opener，将保存了cookie的http处理器，还有设置一个handler用于处理http的URL的打开
        self.opener = urllib.request.build_opener(self.cookie_processor,
                                                  urllib.request.HTTPHandler)  # 将包含了cookie、http处理器、http的handler的资源和urllib2对象绑定在一起
        urllib.request.install_opener(self.opener)

    def doPost(self, text):
        print("正在请求文心...")
        PostData = {
            "str": text
        }
        PostData = urllib.parse.urlencode(PostData).encode("utf-8")
        request = urllib.request.Request('http://ccpl.psych.ac.cn/textmind/analysis', headers=headers)
        with urllib.request.urlopen(request, data=PostData) as f:
            resp = f.read()
            print(resp)
            return resp

    def parse_textmind_feature(self, json_str):
        feature_list = []
        json_dict = json.loads(json_str)
        print(json_dict)
        if json_dict['status'] == 'success':
            result_list = json_dict['result']
            for elem in result_list:
                name = elem['name']
                value = elem['value']
                feature_list.append(value)
        else:
            raise ValueError('文心系统分析返回数据异常')
        return feature_list

    def save_arr(self, filename, X_sp):
        """
        特征向量保存
        """
        np.save(filename, X_sp)
        print("*****************write done over *****************")

    def textmind_action(self, train_lines, test_lines):
        """
        输入文本[] 保存特征
        :param train_lines:
        :param test_lines:
        :return:
        """
        X_train, y_train = self.get_input_output(train_lines)
        X_test, y_test = self.get_input_output(test_lines)

        textmind_train_vec_dm = "textmind_train_vec.npy"
        textmind_train_label_dm = "train_label.npy"
        textmind_test_vec_dm = "textmind_test_vec.npy"
        textmind_test_label_dm = "test_label.npy"

        self.save_arr(textmind_train_vec_dm, np.array(X_train))
        self.save_arr(textmind_train_label_dm, np.array(y_train))
        self.save_arr(textmind_test_vec_dm, np.array(X_test))
        self.save_arr(textmind_test_label_dm, np.array(y_test))

    def get_input_output(self, lines):
        """
        输入文本的lines 返回每行对应的文心特征 和 对应的标签
        :param lines:
        :return:
        """
        list_input_feature = []
        list_output_tag = []
        for t_line in lines:
            temp = t_line.split()
            user_name = temp[0]
            tags = temp[1:6]
            query = temp[6:]
            print(query)
            query = " ".join(query).strip().replace("\n", "")
            json_str = self.doPost(query)
            feature_list = self.parse_textmind_feature(json_str)
            list_input_feature.append(feature_list)

            list_tag = []
            for tag in tags:
                j = int(tag)
                list_tag.append(j)
            list_output_tag.append(list_tag)
        return list_input_feature, list_output_tag


if __name__ == '__main__':
    craw = Crawler()
    train_lines, test_lines = input_textmind_data.load_corpus()
    craw.textmind_action(train_lines, test_lines)
