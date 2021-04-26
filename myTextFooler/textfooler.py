#!/usr/bin/env python
# coding=UTF-8
"""
@Author: Yin Zixin
@LastEditors: Yin Zixin
@Description: 
@Date: 2021-04-25 00:34:00
@LastEditTime: 2021-04-25 00:34:00
"""
import numpy as np
import torch
from torch.autograd import Variable

from .attack import Attack
from .criteria import criteria
import tqdm 

from .USE import USE
import os


class TextFooler(Attack):
    def __init__(self, model=None, device=None, IsTargeted=None, **kwargs):
        """
        @description: Carlini and Wagner’s Attack (C&W)
        @param {
            model:
            device:
            kwargs:
        } 
        @return: None
        """
        super(TextFooler, self).__init__(model, device, IsTargeted)

        self._parse_params(**kwargs)

    def _parse_params(self, **kwargs):
        """
        @description: 
        @param {
            kappa:
            lr:
            init_const:
            lower_bound:
            upper_bound:
            binary_search_steps:
        } 
        @return: None
        """

        self.perturb_ratio = float(kwargs.get("perturb_ratio", 0.))
        self.sim_score_threshold = float(kwargs.get("sim_score_threshold", 0.7))
        self.import_score_threshold = float(kwargs.get("import_score_threshold", -1.))
        self.sim_score_window = int(kwargs.get("sim_score_window", 15))
        self.synonym_num = int(kwargs.get("synonym_num", 50))
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.counter_fitting_embeddings_path = kwargs.get("counter_fitting_embeddings_path", None)
        self.counter_fitting_cos_sim_path = kwargs.get("counter_fitting_cos_sim_path", None)
        self.USE_model_path = str(kwargs.get('USE_model_path', "https://tfhub.dev/google/universal-sentence-encoder-large/3"))
        
        # print(self.counter_fitting_embeddings_path)
        
        
        if self.counter_fitting_embeddings_path is None:
            
            now_path = os.path.split(os.path.abspath(__file__))[0]
            # print(now_path)
            self.counter_fitting_embeddings_path = os.path.join(now_path, 'counter-fitted-vectors.txt')
        
    def pick_most_similar_words_batch(self, src_words, sim_mat, idx2word, ret_count=10, threshold=0.):
        """
        embeddings is a matrix with (d, vocab_size)
        """
        sim_order = np.argsort(-sim_mat[src_words, :])[:, 1:1 + ret_count]
        sim_words, sim_values = [], []
        for idx, src_word in enumerate(src_words):
            sim_value = sim_mat[src_word][sim_order[idx]]
            mask = sim_value >= threshold
            sim_word, sim_value = sim_order[idx][mask], sim_value[mask]
            sim_word = [idx2word[id] for id in sim_word]
            sim_words.append(sim_word)
            sim_values.append(sim_value)
        return sim_words, sim_values

    def attack(self, text_ls, true_label, predictor, stop_words_set, word2idx, idx2word, cos_sim, sim_predictor=None,
            import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15, synonym_num=50,
            batch_size=32):
        # first check the prediction of the original text
        orig_probs = predictor([text_ls]).squeeze()
        orig_label = torch.argmax(orig_probs)
        orig_prob = orig_probs.max()
        if true_label != orig_label:
            return '', 0, orig_label, orig_label, 0
        else:
            len_text = len(text_ls)
            if len_text < sim_score_window:
                sim_score_threshold = 0.1  # shut down the similarity thresholding function
            half_sim_score_window = (sim_score_window - 1) // 2
            num_queries = 1

            # get the pos and verb tense info
            pos_ls = criteria.get_pos(text_ls)

            # get importance score
            leave_1_texts = [text_ls[:ii] + ['<oov>'] + text_ls[min(ii + 1, len_text):] for ii in range(len_text)]
            leave_1_probs = predictor(leave_1_texts, batch_size=batch_size)
            num_queries += len(leave_1_texts)
            leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)
            import_scores = (orig_prob - leave_1_probs[:, orig_label] + (leave_1_probs_argmax != orig_label).float() * (
                        leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0,
                                                                        leave_1_probs_argmax))).data.cpu().numpy()

            # get words to perturb ranked by importance scorefor word in words_perturb
            words_perturb = []
            for idx, score in sorted(enumerate(import_scores), key=lambda x: x[1], reverse=True):
                try:
                    if score > import_score_threshold and text_ls[idx] not in stop_words_set:
                        words_perturb.append((idx, text_ls[idx]))
                except:
                    print(idx, len(text_ls), import_scores.shape, text_ls, len(leave_1_texts))

            # find synonyms
            words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
            synonym_words, _ = self.pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
            synonyms_all = []
            for idx, word in words_perturb:
                if word in word2idx:
                    synonyms = synonym_words.pop(0)
                    if synonyms:
                        synonyms_all.append((idx, synonyms))

            # start replacing and attacking
            text_prime = text_ls[:]
            text_cache = text_prime[:]
            num_changed = 0
            for idx, synonyms in synonyms_all:
                new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
                new_probs = predictor(new_texts, batch_size=batch_size)

                # compute semantic similarity
                if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                    text_range_min = idx - half_sim_score_window
                    text_range_max = idx + half_sim_score_window + 1
                elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                    text_range_min = 0
                    text_range_max = sim_score_window
                elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                    text_range_min = len_text - sim_score_window
                    text_range_max = len_text
                else:
                    text_range_min = 0
                    text_range_max = len_text
                semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                        list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

                num_queries += len(new_texts)
                if len(new_probs.shape) < 2:
                    new_probs = new_probs.unsqueeze(0)
                new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
                # prevent bad synonyms
                new_probs_mask *= (semantic_sims >= sim_score_threshold)
                # prevent incompatible pos
                synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
                pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
                new_probs_mask *= pos_mask

                if np.sum(new_probs_mask) > 0:
                    text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                    num_changed += 1
                    break
                else:
                    new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                            (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                    new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                    if new_label_prob_min < orig_prob:
                        text_prime[idx] = synonyms[new_label_prob_argmin]
                        num_changed += 1
                text_cache = text_prime[:]
            return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


    def random_attack(self, text_ls, true_label, predictor, perturb_ratio, stop_words_set, word2idx, idx2word, cos_sim,
                    sim_predictor=None, import_score_threshold=-1., sim_score_threshold=0.5, sim_score_window=15,
                    synonym_num=50, batch_size=32):
        # first check the prediction of the original text
        orig_probs = predictor([text_ls]).squeeze()
        orig_label = torch.argmax(orig_probs)
        orig_prob = orig_probs.max()
        if true_label != orig_label:
            return '', 0, orig_label, orig_label, 0
        else:
            len_text = len(text_ls)
            if len_text < sim_score_window:
                sim_score_threshold = 0.1  # shut down the similarity thresholding function
            half_sim_score_window = (sim_score_window - 1) // 2
            num_queries = 1

            # get the pos and verb tense info
            pos_ls = criteria.get_pos(text_ls)

            # randomly get perturbed words
            perturb_idxes = random.sample(range(len_text), int(len_text * perturb_ratio))
            words_perturb = [(idx, text_ls[idx]) for idx in perturb_idxes]

            # find synonyms
            words_perturb_idx = [word2idx[word] for idx, word in words_perturb if word in word2idx]
            synonym_words, _ = self.pick_most_similar_words_batch(words_perturb_idx, cos_sim, idx2word, synonym_num, 0.5)
            synonyms_all = []
            for idx, word in words_perturb:
                if word in word2idx:
                    synonyms = synonym_words.pop(0)
                    if synonyms:
                        synonyms_all.append((idx, synonyms))

            # start replacing and attacking
            text_prime = text_ls[:]
            text_cache = text_prime[:]
            num_changed = 0
            for idx, synonyms in synonyms_all:
                new_texts = [text_prime[:idx] + [synonym] + text_prime[min(idx + 1, len_text):] for synonym in synonyms]
                new_probs = predictor(new_texts, batch_size=batch_size)

                # compute semantic similarity
                if idx >= half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                    text_range_min = idx - half_sim_score_window
                    text_range_max = idx + half_sim_score_window + 1
                elif idx < half_sim_score_window and len_text - idx - 1 >= half_sim_score_window:
                    text_range_min = 0
                    text_range_max = sim_score_window
                elif idx >= half_sim_score_window and len_text - idx - 1 < half_sim_score_window:
                    text_range_min = len_text - sim_score_window
                    text_range_max = len_text
                else:
                    text_range_min = 0
                    text_range_max = len_text
                semantic_sims = \
                sim_predictor.semantic_sim([' '.join(text_cache[text_range_min:text_range_max])] * len(new_texts),
                                        list(map(lambda x: ' '.join(x[text_range_min:text_range_max]), new_texts)))[0]

                num_queries += len(new_texts)
                if len(new_probs.shape) < 2:
                    new_probs = new_probs.unsqueeze(0)
                new_probs_mask = (orig_label != torch.argmax(new_probs, dim=-1)).data.cpu().numpy()
                # prevent bad synonyms
                new_probs_mask *= (semantic_sims >= sim_score_threshold)
                # prevent incompatible pos
                synonyms_pos_ls = [criteria.get_pos(new_text[max(idx - 4, 0):idx + 5])[min(4, idx)]
                                if len(new_text) > 10 else criteria.get_pos(new_text)[idx] for new_text in new_texts]
                pos_mask = np.array(criteria.pos_filter(pos_ls[idx], synonyms_pos_ls))
                new_probs_mask *= pos_mask

                if np.sum(new_probs_mask) > 0:
                    text_prime[idx] = synonyms[(new_probs_mask * semantic_sims).argmax()]
                    num_changed += 1
                    break
                else:
                    new_label_probs = new_probs[:, orig_label] + torch.from_numpy(
                            (semantic_sims < sim_score_threshold) + (1 - pos_mask).astype(float)).float().cuda()
                    new_label_prob_min, new_label_prob_argmin = torch.min(new_label_probs, dim=-1)
                    if new_label_prob_min < orig_prob:
                        text_prime[idx] = synonyms[new_label_prob_argmin]
                        num_changed += 1
                text_cache = text_prime[:]
            return ' '.join(text_prime), num_changed, orig_label, torch.argmax(predictor([text_prime])), num_queries


    def generate(self, xs=None, ys=None):
        """
        @description: 
        @param {
            xs:
            ys:
        } 
        @return: adv_xs
        """
        
        adv_xs = []

        idx2word = {}
        word2idx = {}

        print("Building vocab...")
        with open(self.counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1

        if self.counter_fitting_cos_sim_path:
            # load pre-computed cosine similarity matrix if provided
            print('Load pre-computed cosine similarity matrix from {}'.format(self.counter_fitting_cos_sim_path))
            cos_sim = np.load(self.counter_fitting_cos_sim_path)
        else:
            # calculate the cosine similarity matrix
            print('Start computing the cosine similarity matrix!')
            embeddings = []
            with open(self.counter_fitting_embeddings_path, 'r') as ifile:
                for line in ifile:
                    embedding = [float(num) for num in line.strip().split()[1:]]
                    embeddings.append(embedding)
            embeddings = np.array(embeddings)
            product = np.dot(embeddings, embeddings.T)
            norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
            cos_sim = product / np.dot(norm, norm.T)
        print("Cos sim import finished!")

        use = USE(self.USE_model_path)

        stop_words_set = criteria.get_stopwords()

        for idx, (text, true_label) in enumerate(tqdm.tqdm(list(zip(xs, ys)))):
            if self.perturb_ratio > 0.:
                new_text, num_changed, orig_label, \
                new_label, num_queries = self.random_attack(text, true_label, self.model, self.perturb_ratio, stop_words_set,
                                                        word2idx, idx2word, cos_sim, sim_predictor=use,
                                                        sim_score_threshold=self.sim_score_threshold,
                                                        import_score_threshold=self.import_score_threshold,
                                                        sim_score_window=self.sim_score_window,
                                                        synonym_num=self.synonym_num,
                                                        batch_size=self.batch_size)
            else:
                new_text, num_changed, orig_label, \
                new_label, num_queries = self.attack(text, true_label, self.model, stop_words_set,
                                                word2idx, idx2word, cos_sim, sim_predictor=use,
                                                sim_score_threshold=self.sim_score_threshold,
                                                import_score_threshold=self.import_score_threshold,
                                                sim_score_window=self.sim_score_window,
                                                synonym_num=self.synonym_num,
                                                batch_size=self.batch_size)
            # print(new_text)
            adv_xs.append(new_text)
        return adv_xs
