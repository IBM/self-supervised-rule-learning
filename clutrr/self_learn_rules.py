import json
import random
import re
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


class Graph:
    def __init__(self):
        self.adjacency_list = defaultdict(list)

    def add_edge(self, src, pred, dest):
        self.adjacency_list[src].append((pred, dest))

    def _get_paths_between_entities(self, entity1, entity2, hops, visited):
        if entity1 == entity2:
            return [(entity1,)]
        if hops == 0:
            return []
        visited.add(entity1)
        res = []
        for pred, entity in self.adjacency_list[entity1]:
            if entity not in visited:
                for path in self._get_paths_between_entities(entity, entity2, hops - 1, visited):
                    res.append((entity1, pred) + path)
        return res

    def get_paths_between_entities(self, entity1, entity2, hops=0):
        return self._get_paths_between_entities(entity1, entity2, hops + 1, set())


class Data:
    def __init__(self, data_path, batch_size=1, train=False, add_inverses=True):
        self.id2pred, self.pred2id = dict(), dict()
        self.id2ent, self.ent2id = dict(), dict()
        self.predicates = ['son', 'father', 'husband', 'brother', 'grandson', 'grandfather', 'son-in-law',
                           'father-in-law', 'brother-in-law', 'uncle', 'nephew', 'daughter', 'mother', 'wife', 'sister',
                           'granddaughter', 'grandmother', 'daughter-in-law', 'mother-in-law', 'sister-in-law', 'aunt',
                           'niece']
        self.add_predicates()
        print('Reading and preparing data...')
        with open(data_path, 'r') as fp:
            self.data = [data_i for data_i in json.load(fp) if train == False or data_i['task_name'].endswith('.2')]
            fp.close()
        if add_inverses:
            self.expand_data()
        self.data_triple = list(zip(self.get_queries(), self.get_facts()))
        self.data_index = [i for i in range(len(self.data_triple))]
        # self.randomize_data()
        self.batch_size = batch_size
        self.batch_info = self.get_batch_info()
        print(' ...done')

    def expand_data(self):
        expanded_data = []
        for data_i in self.data:
            new_data = deepcopy(data_i)
            new_data['graph_story'][0], new_data['graph_query'] = new_data['graph_query'], new_data['graph_story'][0]
            expanded_data.append(new_data)

            new_data = deepcopy(data_i)
            new_data['graph_story'][1], new_data['graph_query'] = new_data['graph_query'], new_data['graph_story'][1]
            expanded_data.append(new_data)
        self.data.extend(expanded_data)

    def add_predicates(self):
        for predicate in self.predicates:
            self.get_predicate_id(predicate + '+1')
            self.get_predicate_id(predicate + '-1')

    def get_facts(self):
        facts = []
        for data_i in self.data:
            graph = Graph()
            for fact in data_i['graph_story']:
                pred, se, de = self.parse_predicate(fact)
                graph.add_edge(self.get_entity_id(se), self.get_predicate_id(pred + '+1'), self.get_entity_id(de))
                graph.add_edge(self.get_entity_id(de), self.get_predicate_id(pred + '-1'), self.get_entity_id(se))
            facts.append(graph)
        return facts

    def get_queries(self):
        queries = []
        for data_i in self.data:
            pred, se, de = self.parse_predicate(data_i['graph_query'])
            pred += '+1'
            queries.append((self.get_predicate_id(pred), self.get_entity_id(se), self.get_entity_id(de)))
        return queries

    def parse_predicate(self, predicate_string):
        return re.match(r'(.*)\((.*), (.*)\)$', predicate_string).groups()

    def get_batch_info(self):
        n_batches = len(self.data_triple) // self.batch_size
        remaining = len(self.data_triple) % self.batch_size
        extra_per_batch = remaining // n_batches
        left_out = remaining % n_batches
        batch = []
        start = 0
        for batch_i in range(n_batches):
            end = start + self.batch_size + extra_per_batch
            if left_out > 0:
                end += 1
                left_out -= 1
            batch.append((start, end - 1))
            start = end
        return batch

    def randomize_data(self):
        lst = [i for i in range(len(self.data_triple))]
        random.shuffle(lst)
        self.data_index = lst

    def get_data_len(self):
        return len(self.data_triple)

    def get_batches_count(self):
        return len(self.batch_info)

    def get_batch_data(self, batch_i):
        start, end = self.batch_info[batch_i]
        return [self.data_triple[self.data_index[i]] for i in range(start, end + 1)]

    def get_predicate_id(self, pred):
        if pred not in self.pred2id:
            id = len(self.pred2id)
            self.pred2id[pred] = id
            self.id2pred[id] = pred
        return self.pred2id[pred]

    def get_entity_id(self, ent):
        if ent not in self.ent2id:
            id = len(self.ent2id)
            self.ent2id[ent] = id
            self.id2ent[id] = ent
        return self.ent2id[ent]

    def preds_count(self):
        return len(self.pred2id)

    def paths_count(self):
        zero_hop_paths_count = self.preds_count()
        one_hop_paths_count = self.preds_count() ** 2
        return zero_hop_paths_count + one_hop_paths_count

    def get_zero_hop_path_id(self, p):
        return p

    def get_one_hop_path_id(self, p1, p2):
        return self.preds_count() + (p1 * self.preds_count() + p2)

    def get_path_from_id(self, id):
        if id < self.preds_count():
            return self.id2pred[id]
        else:
            id -= self.preds_count()
            p1 = self.id2pred[id // self.preds_count()]
            p2 = self.id2pred[id % self.preds_count()]
            return p1 + ':' + p2


class Rules(nn.Module):
    def __init__(self, rules_count, rule_len):
        super(Rules, self).__init__()
        self.rules_count = rules_count
        self.rules = nn.Embedding(rules_count, rule_len)
        self.rules.weight = torch.nn.Parameter(torch.zeros(rules_count, rule_len))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp, tar, rules_lst):
        # rules model forward pass
        wt_loss = self.param_values_loss()
        est, pred_loss = self.pred_mask_loss(inp, tar, rules_lst)
        return est, pred_loss, wt_loss

    def pred_mask_loss(self, inp, tar, rules_lst):
        # specific method to compute loss when predicate is masked
        rules_t = self.rules(rules_lst)
        est = self.sigmoid(torch.sum(inp * rules_t, dim=1))
        return est, -torch.sum(tar * torch.log(est))

    def param_values_loss(self):
        # loss measuring individual weight values staying between range 0 to 1
        t1 = torch.max(torch.tensor([0.0]), -self.rules.weight)
        t2 = torch.max(torch.tensor([0.0]), (self.rules.weight - torch.tensor([1.0])))
        t3 = torch.max(t1, t2)
        return torch.sum(t3 * t3)


class SelfRuleLearner:
    def __init__(self, data):
        self.data = data
        self.rules_count = self.data.preds_count()
        self.rules_len = self.data.paths_count()
        self.rules = Rules(self.rules_count, self.rules_len)
        self.rules_model_file_path = 'rules_model'
        self.rules_info_file_path = 'rules_info'
        self.view_rules_file_path = 'view_rules'

    def self_learn(self, n_iters, lr=0.0001):
        # main training loop for one of the training schemes. will include all the training schemes
        print('Training Rules Model...')
        torch.set_printoptions(profile="full")
        rules_opt = torch.optim.Adam(self.rules.parameters(), lr=lr)
        for iter_i in range(n_iters):  # for each iteration
            overall_loss = 0.0
            total_prediction_loss, total_weight_loss, total_logical_loss = 0.0, 0.0, 0.0
            for batch_i in range(self.data.get_batches_count()):
                for data_i in self.data.get_batch_data(batch_i):  # for each data item in the batch
                    triple, facts = data_i

                    pp, se, de = triple
                    paths = facts.get_paths_between_entities(se, de, 1)
                    tensors = self.prepare_train_data_pred_mask(triple, paths)
                    if tensors is None:
                        continue
                    inp, tar, rules_lst = tensors
                    est, pred_loss, wt_loss = self.rules(inp, tar, rules_lst)
                    total_prediction_loss += pred_loss.item()
                    total_weight_loss += wt_loss.item()

                    loss = pred_loss + wt_loss
                    overall_loss += loss.item()
                    loss.backward()

                    rules_opt.step()
                    rules_opt.zero_grad()

            print(f'iter={iter_i}, loss={overall_loss / self.data.get_data_len()}')
            # print(f'iter={iter_i}, prediction loss={total_prediction_loss / self.data.get_data_len()}')
            # print(f'iter={iter_i}, weight loss={total_weight_loss / self.data.get_data_len()}')
        print(' ... done')
        self.save_rules(self.rules_model_file_path)
        self.save_rules_info(self.rules_info_file_path)
        self.view_rules(self.view_rules_file_path)

    def prepare_train_data_pred_mask(self, triple, paths):
        # mask pred
        # evaluate all the rules for the candidate paths
        # output: vector of triple validity for all the wikidata predicates, target rule model output
        pp, se, de = triple
        c_wp_lst = set([])
        for path in paths:
            # Zero Hop path
            if len(path) == 3:
                _, p, _ = path
                c_wp = self.data.get_zero_hop_path_id(p)
                if c_wp not in c_wp_lst:
                    c_wp_lst.add(c_wp)
            # One Hop Path
            elif len(path) == 5:
                _, p1, _, p2, _ = path
                c_wp = self.data.get_one_hop_path_id(p1, p2)
                if c_wp not in c_wp_lst:
                    c_wp_lst.add(c_wp)
        if len(c_wp_lst) <= 0:
            return None
        inp = np.zeros(self.rules_len)
        for c_wp in c_wp_lst:
            inp[c_wp] = 1.0
        tar = np.zeros(self.rules_count)
        tar[pp] = 1.0
        inp = torch.Tensor(inp)
        tar = torch.Tensor(tar)
        rules_lst = torch.LongTensor([i for i in range(self.rules_count)])
        return inp, tar, rules_lst

    def save_rules(self, rules_model_file_path):
        print('Writing down rules model into file:', rules_model_file_path)
        torch.save(self.rules, rules_model_file_path)

    def save_rules_info(self, rules_info_file_path):
        print('Writing down rules info into file:', rules_info_file_path)
        rules_info = {'rules-count': self.rules_count, 'rules-len': self.rules_len, 'rules-id-map': self.data.id2pred}
        with open(rules_info_file_path, 'w') as fp:
            json.dump(rules_info, fp, indent=4)
            fp.close()

    def view_rules(self, view_rules_file_path):
        print('Writing down learned rules into file:', view_rules_file_path)
        rules = {}
        for i in range(self.rules_count):
            pp = self.data.id2pred[i]
            rule_weights = self.rules.rules(torch.LongTensor([i]))
            rule = []
            for j in range(self.rules_len):
                wp = self.data.get_path_from_id(j)
                weight = rule_weights[0][j].item()
                if weight > 0.5:
                    rule.append((wp, weight))
            rules[pp] = rule
        with open(view_rules_file_path, 'w') as fp:
            json.dump(rules, fp, indent=4)
            fp.close()


class RulesEvaluation:
    def __init__(self, data):
        self.data = data
        self.srl = SelfRuleLearner(self.data)
        self.rules = self.load_rules(self.srl.rules_model_file_path)
        self.rules_count, self.rules_len, self.id2rule = self.load_rules_info(self.srl.rules_info_file_path)

    def load_rules(self, rules_model_file_path):
        print('Reading rules model from file:', rules_model_file_path)
        return torch.load(rules_model_file_path)

    def load_rules_info(self, rules_info_file_path):
        print('Reading rules info from file:', rules_info_file_path)
        with open(rules_info_file_path, 'r') as fp:
            rules_info = json.load(fp)
            rules_count = rules_info['rules-count']
            rules_len = rules_info['rules-len']
            id2rule = rules_info['rules-id-map']
            fp.close()
            return rules_count, rules_len, id2rule

    def get_prediction(self, triple, paths):
        tensors = self.srl.prepare_train_data_pred_mask(triple, paths)
        try:
            if tensors is not None:
                inp, tar, rules_lst = tensors
                est, _, _ = self.rules(inp, tar, rules_lst)
                return torch.max(est).item(), int((torch.argmax(est)).item())
            else:
                return 0, None
        except Exception as e:
            print(e)

    @lru_cache(None)
    def infer(self, path, triple):
        # zero Hop Path
        if len(path) == 3:
            return 1, path[1]
        else:
            res = (0, None)
            for i in range(2, len(path) - 2, 2):
                s_i, r_i = self.infer(path[:i + 1], triple)
                si_, ri_ = self.infer(path[i:], triple)
                s, r = self.get_prediction(triple, [(path[0], r_i, path[i], ri_, path[-1])])
                res = max(res, (s_i * s * si_, r))
            return res

    def print_paths(self, facts, se, de):
        for path in facts.get_paths_between_entities(se, de, 9):
            for ind, id in enumerate(path):
                print(self.id2rule[str(id)] if ind % 2 else self.data.id2ent[id], end='  ')
            print()

    def eval(self):
        print('Evaluating rules model...')
        correct_count = 0
        for batch_i in range(self.data.get_batches_count()):
            for data_i in self.data.get_batch_data(batch_i):
                triple, facts = data_i
                pp, se, de = triple
                best = (0, None)
                for path in facts.get_paths_between_entities(se, de, 9):
                    best = max(best, self.infer(path, triple))

                tar_rule = self.id2rule[str(pp)]
                est_rule = self.id2rule.get(str(best[1]))
                result = est_rule == tar_rule

                print('Path=', end=' ')
                self.print_paths(facts, se, de)
                print(f'estimated_rule={est_rule}, target_rule={tar_rule}, Result={result}')
                correct_count += result
        accuracy = correct_count * 100.0 / self.data.get_data_len()
        print(f'correct_count={correct_count}, total_count={self.data.get_data_len()}, overall_accuracy={accuracy}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_data_json_path", type=str)
    parser.add_argument("--test_data_json_path", type=str)
    args = parser.parse_args()

    # train rules model
    train_data = Data(args.train_data_json_path, train=True, add_inverses=False)
    srl = SelfRuleLearner(train_data)
    srl.self_learn(5, 0.01)

    # eval rules model
    eval_data = Data(args.test_data_json_path, train=False, add_inverses=False)
    rules_eval = RulesEvaluation(eval_data)
    rules_eval.eval()
