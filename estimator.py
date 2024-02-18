import math
import pdb

import pandas as pd


class NDCG():
    def __init__(self):
        self.result = []
        self.name = "NDCG"


    def calculate_NDCG(self, dictionary, date):
        """
        dictionary:
            code1:
                x:
                y:
                output:
                gain: need add
                eta_output: need add
                eta_label: need add
            code2:

        :param dictionary:
        :return:
        """
        output_set = sorted(dictionary.items(), key=lambda dic: dic[1]["output"], reverse=True)
        output_codes = [item[0] for item in output_set]
        label_set = sorted(dictionary.items(), key=lambda dic: dic[1]["y"], reverse=True)
        label_codes = [item[0] for item in label_set]

        # add G and eta_output, calculate DCG
        DCG = 0
        for i, code in enumerate(output_codes):
            label = dictionary[code]["y"].item()
            dictionary[code]["gain"] = pow(2, label) - 1
            dictionary[code]["eta_output"] = 1 / math.log(i + 2, 2)
            DCG += dictionary[code]["gain"] * dictionary[code]["eta_output"]

        # add eta_label, calculate Zk
        Zk = 0
        for i, code in enumerate(label_codes):
            dictionary[code]["eta_label"] = 1 / math.log(i + 2, 2)
            Zk += dictionary[code]["gain"] * dictionary[code]["eta_label"]

        NDCG = DCG / Zk

        self.result.append([date, NDCG])

    def sum_score(self):
        score_sum = 0
        for date, score in self.result:
            score_sum += score
        return score_sum

    def mean_score(self):
        return self.sum_score() / len(self.result)

    def save_result(self, save_path):
        data = pd.DataFrame(data=self.result)
        data.to_csv(save_path)
        print(f"save test result in: {save_path}")

    def _weight1(self, x):
        if x >= 0:
            out = math.log(3 * x + 2, math.e) + 0.5
        else:
            out = 0.5 * (math.log(-3 * x + 2, math.e) + 0.5)
        return out

    def _weight2(self, x):
        out = 0.5 - pow(x / 15, 2)
        return out

    def weight(self, x):
        out = self._weight1(x) + self._weight2(x)
        return out


class MyEstimator():
    def __init__(self):
        self.result = []
        self.name = "MyEstimator"

    def calculate(self, dictionary, date):
        length = int(len(dictionary) * 0.2 + 1)

        output_set = sorted(dictionary.items(), key=lambda dic: dic[1]["output"], reverse=True)
        output_codes = [item[0] for item in output_set][0:length]
        label_set = sorted(dictionary.items(), key=lambda dic: dic[1]["y"], reverse=True)
        label_codes = [item[0] for item in label_set][0:length]

        number = 0
        for output_code in output_codes:
            if output_code in label_codes:
                number += 1

        self.result.append([date, number / length])

    def save_result(self, save_path):
        data = pd.DataFrame(data=self.result)
        data.to_csv(save_path)
        print(f"save test result in: {save_path}")

    def sum_score(self):
        score_sum = 0
        for date, score in self.result:
            score_sum += score
        return score_sum

    def mean_score(self):
        return self.sum_score() / len(self.result)

