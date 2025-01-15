import re
import json
import pandas
import os

"""

基于脚本的多轮对话系统

"""


class DialogueSystem:
    def __init__(self):
        self.node_id_to_node_info = {}
        self.slot_info = {}
        self.load()

    def load(self):
        self.load_scenario("scenario/scenario-买衣服.json")
        self.slot_template("scenario/slot_fitting_template.xlsx")

    def load_scenario(self, path):
        scenario_name = os.path.basename(path).split(".")[0]
        with open(path, "r", encoding="utf-8") as f:
            scenario = json.load(f)
        for node in scenario:
            node_id = scenario_name + "-" + node["id"]
            if "childnode" in node:
                node["childnode"] = [scenario_name + "-" + child for child in node["childnode"]]
            self.node_id_to_node_info[node_id] = node

    def slot_template(self, path):
        df = pandas.read_excel(path)
        for index, row in df.iterrows():
            slot = row["slot"]
            query = row["query"]
            value = row["values"]
            self.slot_info[slot] = {"query": query, "values": value}

    def get_response(self, user_input, memory):
        memory["user_input"] = user_input
        memory = self.nlu(memory)
        memory = self.dst(memory)
        memory = self.policy(memory)
        memory = self.nlg(memory)
        return memory["bot_response"], memory

    def nlu(self, memory):
        memory = self.get_intent(memory)
        memory = self.get_slot(memory)
        return memory

    def get_intent(self, memory):
        max_score = -1
        hit_intent = None
        for node in memory["available_nodes"]:
            node = self.node_id_to_node_info[node]
            score = self.get_node_score(node, memory)
            if score > max_score:
                max_score = score
                hit_intent = node
        memory["hit_intent"] = hit_intent
        memory["hit_intent_score"] = max_score
        repeat_score = self.get_sentence_similarity(memory["user_input"], "重复")
        if repeat_score > max_score:
            memory["init_intent"] = hit_intent["intent"]  # 记录初始意图
            memory["hit_intent"]["intent"] = "重复"
            memory["hit_intent_score"] = repeat_score
        return memory

    def get_node_score(self, node, memory):
        intent = memory["user_input"]
        node_intents = node["intent"]
        scores = []
        for node_intent in node_intents:
            sentence_similiarity = self.get_sentence_similarity(intent, node_intent)
            scores.append(sentence_similiarity)
        return max(scores)

    def get_sentence_similarity(self, sentence1, sentence2):
        # TODO: 实现句子相似度计算
        set1 = set(sentence1)
        set2 = set(sentence2)
        return len(set1 & set2) / len(set1 | set2)

    def get_slot(self, memory):
        """槽位抽取"""
        hit_intent = memory["hit_intent"]
        if "slot" in hit_intent.keys():
            for slot in hit_intent["slot"]:
                values = self.slot_info[slot]["values"]
                if re.search(values, memory["user_input"]):
                    memory[slot] = re.search(values, memory["user_input"]).group()
        return memory

    def dst(self, memory):
        # TODO: 实现对话状态跟踪, 判断当前intent所需的槽位是否已经被填满
        hit_intent = memory["hit_intent"]
        if "slot" in hit_intent.keys():
            for slot in hit_intent["slot"]:
                if slot not in memory.keys():
                    memory["need_slot"] = slot
                    return memory
        memory["need_slot"] = None
        return memory

    def policy(self, memory):
        # TODO: 实现对话策略
        if memory["hit_intent"]["intent"] == "重复":
            memory["action"] = "repeat"
            memory["available_nodes"] = [node_id for node_id in self.node_id_to_node_info.keys() if
                                         self.node_id_to_node_info[node_id]["id"] == memory["hit_intent"]["id"]]
        elif memory["need_slot"] is None:
            memory["action"] = "answer"
            # 开放子节点
            if "childnode" in memory["hit_intent"].keys():
                memory["available_nodes"] = memory["hit_intent"]["childnode"]
            else:
                memory['status'] = False
        else:
            memory["action"] = "ask"
            # 停留在当前节点
            memory["available_nodes"] = [node_id for node_id in self.node_id_to_node_info.keys() if
                                         self.node_id_to_node_info[node_id]["id"] == memory["hit_intent"]["id"]]
        return memory

    def nlg(self, memory):
        # TODO: 实现自然语言生成
        if memory["action"] == "repeat":  # 重复回答
            if "repeat" in memory.keys():
                memory["bot_response"] = "重复上一轮的问题\n" + memory["repeat"]  # 重复当前对话内容
            else:
                memory["bot_response"] = "抱歉，我无法重复上一轮的问题"
            memory["hit_intent"]["intent"] = memory["init_intent"]  # 恢复初始意图
        elif memory["action"] == "answer":  # 直接回答
            memory["bot_response"] = self.replace_slot(memory["hit_intent"]["response"], memory)
            memory["repeat"] = "      " + memory["bot_response"]  # 记录当前对话内容
        else:  # 反问
            slot = memory["need_slot"]
            query = self.slot_info[slot]["query"]
            memory["bot_response"] = query
            memory["repeat"] = "      " + memory["bot_response"]  # 记录当前对话内容
        return memory

    def replace_slot(self, response, memory):
        if "slot" in memory["hit_intent"].keys():
            slots = memory["hit_intent"]["slot"]
            for slot in slots:
                response = response.replace(slot, memory[slot])
        return response


if __name__ == '__main__':
    ds = DialogueSystem()
    print(ds.node_id_to_node_info)
    print(ds.slot_info)
    memory = {"available_nodes": ["scenario-买衣服-node1"], 'status': True}

    while memory['status']:
        user_input = input("User: ")
        response, memory = ds.get_response(user_input, memory)
        print("Bot: ", response)
        print(memory)
